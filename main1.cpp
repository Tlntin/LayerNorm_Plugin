#include <cuda_runtime.h>
#include <chrono>
#include <NvInfer.h>
#include <NvInferRuntime.h>
#include <NvOnnxParser.h>
#include "plugin/layer_norm_plugin.h"
#include <string>
#include <iostream>
#include <fstream>
#include <xtensor/xarray.hpp>
#include <xtensor/xnpy.hpp>


using namespace nvinfer1;

PluginFieldCollection    LayerNormPluginCreator::fc_ {};
std::vector<PluginField> LayerNormPluginCreator::attr_;
REGISTER_TENSORRT_PLUGIN(LayerNormPluginCreator);


const std::size_t batch = 64;
const std::size_t seq_len = 256;
const std::size_t embedding_size = 768;


class TRTLogger: public nvinfer1::ILogger {
 public:
  void log(Severity severity , nvinfer1::AsciiChar const * msg) noexcept override{
    if (severity == Severity::kERROR) {
      std::cout << "\0x33[31m" << "ERROR " <<
        msg << "\0x33[0m" << std::endl;
    } else if (severity <= Severity::kINFO) {
      std::cout << "\0x33[33m" << "INFO " <<
        msg << "\0x33[0m" << std::endl;
    } 
  }
};


double cpu_time() {
  // 获取当前时间戳(毫秒)
  auto now_time = std::chrono::system_clock::now().time_since_epoch();
  return std::chrono::duration_cast<std::chrono::microseconds>(
    now_time).count() / 1000.0;
}


bool build_engine(std::string onnx_path = "data/model.onnx") {
  /*
  构建TensorRT引擎
  */
  bool result = true;
  TRTLogger logger;
  nvinfer1::IBuilder * builder = nvinfer1::createInferBuilder(logger);
  nvinfer1::IBuilderConfig * config = builder->createBuilderConfig();
  nvinfer1::INetworkDefinition *network = builder->createNetworkV2(1);
  nvinfer1::IOptimizationProfile * profile = \
    builder->createOptimizationProfile();
  nvonnxparser::IParser * parser = nvonnxparser::createParser(
    *network, logger);
  // 后面一个参数是是否显示详情
  bool res1 = parser->parseFromFile(onnx_path.c_str(), 1);
  printf("解析onnx成功，开始生成tensorRT引擎\n");
  if (res1) {
    nvinfer1::ITensor * inputs = network->getInput(0);
    printf("input name %s\n", inputs->getName());
    // printf("input dim is %d\n", inputs->getDimensions());
    // 设置最小，适中，最大维度
    profile->setDimensions(
      inputs->getName(),
      nvinfer1::OptProfileSelector::kMIN,
      nvinfer1::Dims3(1, 1, embedding_size)
    );
    profile->setDimensions(
      inputs->getName(),
      nvinfer1::OptProfileSelector::kOPT,
      nvinfer1::Dims3(batch, seq_len, embedding_size)
    );
    profile->setDimensions(
      inputs->getName(),
      nvinfer1::OptProfileSelector::kMAX,
      nvinfer1::Dims3(batch * 2, 2 * seq_len, embedding_size)
    );
    config->addOptimizationProfile(profile);
    size_t workspace = 4 << 10;
    std::cout << "\033[33m" << "workspace size is " <<
      workspace / 1024.0f << "GB" << "\033[0m" << std::endl;
    config->setMaxWorkspaceSize(workspace);
    config->setProfilingVerbosity(nvinfer1::ProfilingVerbosity::kDETAILED);
    // 构建engine data
    nvinfer1::IHostMemory * engine_data =  builder->buildSerializedNetwork(
      *network, *config);
    if (engine_data != nullptr) {
      FILE * f = fopen("data/model.engine", "wb");
      std::cout << "\033[33m" << "engine_size " << engine_data->size()
        << "\033[3m" << std::endl;
      fwrite(engine_data->data(), 1, engine_data->size(), f);
      fclose(f);
    } else {
      result = false;
    }
    delete engine_data;
  } else {
    std::cout << "\033[31m" << "onnx parser failed " << "\033[0m" << std::endl;
    result = false;
    delete network;
    delete config;
  }
  delete builder;
  return result;
}

bool load_data(std::string file_name, std::vector<char> * v1) {
  // 加载engine数据
  std::ifstream f(file_name, std::ios::binary | std::ios::in);
  f.seekg(0, std::ios::end);
  std::size_t length = f.tellg();
  bool res = true;
  if (length > 0) {
    f.seekg(0, std::ios::beg);
    v1->resize(length);
    std::cout << "length is " << length << "" << std::endl;
    f.read(v1->data(), length);
    res = true;
  } else {
    res = false;
  }
  f.close();
  return res;
}


bool compare_result(float * h_output1, float * h_output2, const std::size_t n) {
  /*
  计算结果对比
  params h_output1: 原始输出
  params h_output2: 插件输出
  params n: 总的输出元素个数
  */
  // 定义最大允许误差值
  float epsilon = 1e-4;
  for (int i = 0; i < n; ++i) {
    if (abs(h_output1[i] - h_output2[i]) > epsilon) {
      return false;
    }
  }
  return true;
}

void infercence() {
  // 开始执行推理模型
  double st0 = cpu_time(); 
  std::string file_name("data/model.engine");
  std::vector<char> model_data;
  bool res1 = load_data(file_name, &model_data);
  if (res1) {
    TRTLogger logger;
    nvinfer1::IRuntime * runtime = nvinfer1::createInferRuntime(logger);
    nvinfer1::ICudaEngine * engine = runtime->deserializeCudaEngine(
      model_data.data(), model_data.size()
    );
    if (engine != nullptr) {
      std::cout << "\033[33m" << "engine load success!" << "\033[0m"
         << std::endl;
      double et0 = cpu_time();
      nvinfer1::IExecutionContext * context = engine->createExecutionContext();
      cudaStream_t stream;
      cudaStreamCreate(&stream);
      // 先构建输入数据
      auto input_data = xt::load_npy<float>("data/input.npy");
      const std::size_t n = batch * seq_len * embedding_size;
      const std::size_t n_bytes = n * sizeof(float);
      float * h_output = static_cast<float *>(malloc(n_bytes));
      float * d_input = nullptr;
      float * d_output = nullptr;
      checkRuntime(
        cudaMalloc(reinterpret_cast<float **>(&d_input), n_bytes));
      checkRuntime(
        cudaMalloc(reinterpret_cast<float **>(&d_output), n_bytes));
      context->setBindingDimensions(
        0, nvinfer1::Dims3(batch, seq_len, embedding_size));
      void * binding[] = {d_input, d_output};
      // 这里应该要一个check宏
      checkRuntime(cudaMemcpyAsync(
        d_input, input_data.data(), n_bytes, cudaMemcpyHostToDevice, stream
      ));
      double st1 = cpu_time();
      context->enqueueV2(
        reinterpret_cast<void * const *>(binding), stream, nullptr);
      checkRuntime(cudaMemcpyAsync(
        h_output, d_output, n_bytes, cudaMemcpyDeviceToHost, stream 
      ));
      checkRuntime(cudaStreamSynchronize(stream));
      double et1 = cpu_time();
      std::cout << "prepare time is " << (et0 - st0) / 1000.0 << "s\n";
      std::cout << "infer time is " << (et1 - st1) << "ms" << std::endl;

      auto out_data = xt::load_npy<float>("data/output.npy");
      // 对比结果
      bool res = compare_result(out_data.data(), h_output, n);
      if (res) {
        std::cout << "\033[33m" << "compare result is True" << 
          "\033[0m" << std::endl;
      } else {
        std::cout << "\031[33m" << "compare result is False" << 
          "\033[0m" << std::endl;
      }
      
      // 释放区
      free(h_output);
      cudaFree(d_input);
      cudaFree(d_output);
      delete context;
      cudaStreamDestroy(stream);
      std::cout << "inference success !" << std::endl;
    }
    delete engine;
    delete runtime;
  } else {
    std::cerr << "engine load failed!" << std::endl;
  }
}


int main() {
  
  // bool res2 = build_engine();
  // std::cout << "build engine result is \033[33m" << res2 
  //   << "\033[0m" << std::endl;
  infercence();
}