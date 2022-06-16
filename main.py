import torch
import numpy as np
import onnx
import onnx_graphsurgeon as gs
import tensorrt as trt
from colored import stylize, fg
from collections import OrderedDict
import onnxruntime as ort
import ctypes
from cuda import cudart
import os


now_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(now_dir, "data")
if not os.path.exists(data_dir):
    os.mkdir(data_dir)
input_path = os.path.join(data_dir, "input.npy")
output_path = os.path.join(data_dir, "output.npy")
onnx_path = os.path.join(data_dir, "model.onnx")
surgeon_onnx_path = os.path.join(data_dir, "surgeon_model.onnx")
plugin_path = os.path.join(now_dir, "plugin", "liblayer_norm_plugin2.so")
engine_path = os.path.join(data_dir, "model.engine")

batch_size = 64
seq_length = 256
embedding_size = 768


class Model(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer_norm = torch.nn.LayerNorm(
            embedding_size, elementwise_affine=True, eps=1e-6)

    def forward(self, x):
        x = torch.mul(x, 1.0)
        x = self.layer_norm(x)
        x = torch.mul(x, 1.0)
        return x


def export_model():
    """
    导出torch模型到onnnx
    """
    print("开始导出pytorch模型到onnx")

    # 获取GPU计算结果
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # input_data = torch.randint(
    #    1, 100, size=(batch_size, seq_length, embedding_size))
    input_data = torch.rand(
        size=(batch_size, seq_length, embedding_size), dtype=torch.float32)
    print("输入文件保存成功")
    np.save(input_path, input_data.cpu().data.numpy().astype(np.float32))
    model = Model().to(device)
    model.eval()
    input_data = input_data.to(device)
    outputs = model(input_data)
    outputs = outputs.cpu().data.numpy().astype(np.float32)
    # print("output is ", outputs)
    print("输出文件保存成功")
    np.save(output_path, outputs)

    # 开始导出模型
    torch.onnx.export(
        model,
        args=(input_data,),
        f=onnx_path,
        input_names=["input"],
        output_names=["outputs"],
        opset_version=13,
        dynamic_axes={
            "input": {0: "batch_size", 1: "seq_length", 2: "embedding"},
        }
    )
    print("模型导出成功")


def is_layer_norm_node(node: gs.Node):
    """
    判断是否为LayerNorm节点
    """
    if len(node.inputs) == 2:
        left_pre = node.i(0)
        right_pre = node.i(1)
        if left_pre.op != "Sub" or right_pre.op != "Sqrt":
            # print("1")
            return False
        if len(right_pre.inputs) == 0 or right_pre.i().op != "Add":
            # print("2")
            return False
        pre_node = right_pre.i()
        if len(pre_node.inputs) == 0 or len(pre_node.inputs[0].inputs) == 0:
            # print("3")
            return False
        if pre_node.i().op != "ReduceMean":
            # print("4")
            return False
        pre_node = pre_node.i()
        if len(pre_node.inputs) == 0 or len(pre_node.inputs[0].inputs) == 0:
            # print("5")
            return False
        if pre_node.i().op != "Pow":
            # print("6")
            return False
        pre_node = pre_node.i()
        if len(pre_node.inputs) == 0 or len(pre_node.inputs[0].inputs) == 0:
            return False
        if pre_node.i().op != "Sub":
            # print("7")
            return False
        pre_node = pre_node.i()
        if len(pre_node.inputs) <= 1 or len(pre_node.inputs[1].inputs) == 0:
            return False
        pre_node = pre_node.i(1)
        if pre_node.op == "ReduceMean":
            # print(8)
            return True
    return False


def replace_onnx():
    """
    修改onnx模型，将中间的LayerNorm层替换为TensorRT插件层
    """
    graph = gs.import_onnx(onnx.load(onnx_path))
    n = 0
    for node in graph.nodes:
        if node.op != "Div":
            continue
        # 判断当前Div是否为LayerNorm
        is_layer_norm = is_layer_norm_node(node)
        if is_layer_norm:
            print("find one layer norm")
            # 属性必须为数组
            epsilon = node.i(1).i().i(1).attrs["value"].values.reshape(1)
            # 新增获取gamma, beta属性
            gamma = node.o().inputs[1].values
            beta = node.o().o().inputs[1].values
            print("epsilon", epsilon)
            temp_outputs = gs.Variable(
                name=f"LayerNorm_output_{n}", dtype=np.float32,
                shape=None
            )
            new_node = gs.Node(
                op="LayerNorm",
                name=f"LayerNorm_{n}",
                attrs=OrderedDict(epsilon=epsilon, gamma=gamma, beta=beta),
                inputs=[node.i(0).inputs[0]],
                outputs=[temp_outputs]
            )
            graph.nodes.append(new_node)
            out_node = node.o().o()
            print("output node is ", out_node)
            # 建立连接
            if len(out_node.outputs) > 0 and \
                    len(out_node.outputs[0].outputs) > 0:
                # out_node.o().inputs[0] = temp_outputs
                out_name = out_node.outputs[0]
                # print("out_name", out_name)
                for sub_node in list(out_node.outputs[0].outputs):
                    # print("sub node", sub_node)
                    for i, input_node in enumerate(sub_node.inputs):
                        if input_node == out_name:
                            print("link node ", i, input_node)
                            sub_node.inputs[i] = temp_outputs
            # 最后一个节点
            else:
                new_node.outputs = out_node.outputs
                raise Exception("请检查节点")
            out_node.outputs.clear()
            n += 1
    graph.cleanup()
    print(graph.outputs)
    onnx.save(gs.export_onnx(graph), surgeon_onnx_path)
    print("surgeon onnx model save success!")


def build_engine():
    """
    构建TensorRT引擎
    """
    print("start build TensorRT engine")
    logger = trt.Logger(trt.Logger.ERROR)
    # 初始化插件
    trt.init_libnvinfer_plugins(logger, "")
    ctypes.cdll.LoadLibrary(plugin_path)
    builder = trt.Builder(logger)
    config = builder.create_builder_config()
    profile = builder.create_optimization_profile()
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    config.max_workspace_size = 3 << 30
    # 解析onnx
    parser = trt.OnnxParser(network, logger)
    with open(surgeon_onnx_path, "rb") as f:
        parser.parse(f.read())
        for i in range(parser.num_errors):
            print(stylize("parser erorr ", fg("red")), parser.get_error(i))
    inputs = network.get_input(0)
    print("output name", network.get_output(0).name)
    profile.set_shape(
        inputs.name,
        min=[1, 1, embedding_size],
        opt=[batch_size, seq_length, embedding_size],
        max=[batch_size * 2, seq_length * 2, embedding_size]
    )
    config.add_optimization_profile(profile)
    engine_data = builder.build_serialized_network(network, config)
    if engine_data is None:
        print(stylize("engine data build failed", fg("red")))
        return
    with open(engine_path, "wb") as f:
        f.write(engine_data)
        print(stylize("build TensorRT Engine success!", fg("yellow")))


def inference():
    """
    正式对模型进行推理
    """
    logger = trt.Logger(trt.Logger.ERROR)
    trt.init_libnvinfer_plugins(logger, "")
    ctypes.cdll.LoadLibrary(plugin_path)
    runtime = trt.Runtime(logger)
    with open(engine_path, "rb") as f:
        engine_data = f.read()
    engine = runtime.deserialize_cuda_engine(engine_data)
    context = engine.create_execution_context()
    context.set_binding_shape(0, [batch_size, seq_length, embedding_size])
    print("=" * 20)
    print("output shape", context.get_binding_shape(1))
    print("=" * 20)

    h_input = np.ascontiguousarray(
        np.load(input_path).reshape(-1))

    h_output = np.empty(
        shape=context.get_binding_shape(1),
        dtype=trt.nptype(engine.get_binding_dtype(1))
    )
    print(h_output.shape)
    print(h_output.dtype)

    res1, d_input = cudart.cudaMalloc(h_input.nbytes)
    if res1 != cudart.cudaError_t.cudaSuccess:
        print(stylize("Error", fg("yellow")), res1)
        return
    res2, d_output = cudart.cudaMalloc(h_output.nbytes)
    if res2 != cudart.cudaError_t.cudaSuccess:
        print(stylize("Error", fg("yellow")), res2)
        return
    """
    res3, stream = cudart.cudaStreamCreate()
    if res3 != cudart.cudaError_t.cudaSuccess:
        print(stylize("Error", fg("yellow")), res3)
        return
    """

    res4 = cudart.cudaMemcpy(
        d_input, h_input.ctypes.data, h_input.nbytes,
        cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
    )
    print(res4)
    context.execute_v2(
        [int(d_input), int(d_output)])
    res5 = cudart.cudaMemcpy(
       h_output.ctypes.data, d_output, h_output.nbytes,
       cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
    )
    print(res5)

    cudart.cudaFree(d_input)
    cudart.cudaFree(d_output)
    return h_output


def check(a, b, weak=False, checkEpsilon=1e-4):
    if weak:
        res = np.all(np.abs(a - b) < checkEpsilon)
    else:
        res = np.all(a == b)
    diff0 = np.max(np.abs(a - b))
    diff1 = np.max(np.abs(a - b) / (np.abs(b) + checkEpsilon))
    print("check:", res, diff0, diff1)


if __name__ == "__main__":
    # 1. 导出模型
    export_model()
    # 2. 替换模型
    replace_onnx()
    # 3. 构建TensorRT Engine
    build_engine()
    # 4. 获取推理结果
    outputs2_1 = inference()

    # # 5. 对比计算结果
    outputs1_1 = np.load(output_path)
    check(outputs2_1, outputs1_1, True)

    print("")
    print("=" * 20)
    print("onnx runtime result")
    session = ort.InferenceSession(onnx_path)
    inputs = np.load(input_path).astype(np.float32)
    outputs = session.run(
        ["outputs"], {"input": inputs})
    print("*" * 20)
    check(outputs[0], outputs1_1, True)
    print("*" * 20)
