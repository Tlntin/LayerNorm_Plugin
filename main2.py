import os
import time
import numpy as np
from polygraphy.backend.trt import (
    EngineFromNetwork, NetworkFromOnnxPath,
    TrtRunner, SaveEngine, CreateConfig, Profile
)
from polygraphy.backend.trt.loader import LoadPlugins


def check(a, b, weak=False, checkEpsilon=1e-4):
    if weak:
        res = np.all(np.abs(a - b) < checkEpsilon)
    else:
        res = np.all(a == b)
    diff0 = np.max(np.abs(a - b))
    diff1 = np.max(np.abs(a - b) / (np.abs(b) + checkEpsilon))
    print("check:", res, diff0, diff1)


now_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(now_dir, "data")
if not os.path.exists(data_dir):
    os.mkdir(data_dir)

input_path = os.path.join(data_dir, "input.npy")
output_path = os.path.join(data_dir, "output.npy")
onnx_path = os.path.join(data_dir, "model.onnx")
surgeon_onnx_path = os.path.join(data_dir, "surgeon_model.onnx")
plugin_path = os.path.join(now_dir, "plugin", "liblayer_norm_plugin2.so")
assert os.path.exists(plugin_path), f"插件路径, {plugin_path}不存在"
engine_path = os.path.join(data_dir, "model.engine")

batch_size = 64
seq_length = 256
embedding_size = 768

plugin = LoadPlugins(plugins=[plugin_path])()
config = CreateConfig(max_workspace_size=4 << 10)
profile = Profile()
profile.add(
    "input",
    min=(1, 1, embedding_size),
    opt=(batch_size, seq_length, embedding_size),
    max=(batch_size * 2, seq_length * 2, embedding_size))
config.profiles = [profile]
input_data = np.load(input_path).astype(np.float32)
print("=" * 20)
print("input shape is", input_data.shape, "input dtype is ", input_data.dtype)
print("=" * 20)

engine = EngineFromNetwork(
    NetworkFromOnnxPath(surgeon_onnx_path), config=config)
engine = SaveEngine(engine=engine, path=engine_path)
output = np.load(output_path).astype(np.float32)
output_dict = {
    "outputs": output,
}
with TrtRunner(engine=engine) as trt:
    outputs = trt.infer(feed_dict={"input": input_data})
    st = time.time()
    outputs = trt.infer(feed_dict={"input": input_data})
    et = time.time()
    # V1版时间 0.00665
    print("infer time during {:.2f}ms".format((et - st) * 1000))
    print(outputs.keys())
    print(len(outputs))
    for k in outputs.keys():
        print(k)
        raw_out = output_dict[k]
        new_out = outputs[k]
        check(raw_out, new_out, True)
