#include <NvInfer.h>
#include <cuda_fp16.h>
#include <string>
#include <vector>
#include <memory>

#define checkRuntime(op)  __check_cuda_runtime((op), #op, __FILE__, __LINE__)

bool __check_cuda_runtime(cudaError_t code, const char* op, const char* file, int line){
  if(code != cudaSuccess){
    const char* err_name = cudaGetErrorName(code);
    const char* err_message = cudaGetErrorString(code);
    printf("runtime error %s:%d  %s failed. \n  code = %s, message = %s\n", file, line, op, err_name, err_message);
    return false;
  }
  return true;
}

// +------- Debug wrapper ------------------------------------------------------
#define YELLOW "\033[33m"
#define RED "\033[31m"
#define NONE "\033[0m"

#ifdef DEBUG
    #define LOG_INFO(op) __log_info(op)
    void __log_info(const char * msg) {
      printf(YELLOW "Log info: %s\n" NONE, msg);
    }

    void __log_info(const std::string msg) {
      printf(YELLOW "Log info: %s\n" NONE, msg.c_str());
    }

    #define WHERE_AM_I()                                 \
        do                                               \
        {                                                \
            printf("[%s]: this=->%p\n", __func__, this); \
        } while (0);
    
    
#else
    #define WHERE_AM_I()
		#define LOG_INFO(op)
#endif // #ifdef DEBUG

#define LOG_ERROR(op) __log_error(op)
void __log_error(const char * msg) {
	printf(RED "Error: %s\n" NONE, msg);
}

void __log_error(std::string msg) {
	printf(RED "Error: %s\n" NONE, msg.c_str());
}



// +------- Plguin -------------------------------------------------------------
namespace
{
static const char *PLUGIN_NAME {"LayerNorm"};
static const char *PLUGIN_VERSION {"1"};
} // namespace

namespace nvinfer1
{


// +------- Plugin body --------------------------------------------------------
class LayerNormPlugin : public IPluginV2DynamicExt
{
 private:
  std::string name_;
  std::string namespace_;

  std::vector<float> h_gamma;
  std::vector<float> h_beta;
  float epsilon_;
  std::size_t length_;

 public:
  LayerNormPlugin(
    const std::string &name, float epsilon, std::size_t length,
    const std::vector<float> gamma, const std::vector<float> beta):
    name_(name), epsilon_(epsilon), length_(length), 
    h_gamma(gamma), h_beta(beta)
  {
    WHERE_AM_I();
		LOG_INFO("init epsilon is " + std::to_string(epsilon_));
		LOG_INFO("init length is " + std::to_string(length_));
		LOG_INFO("init gamma is " + std::to_string(h_gamma[0]));
		LOG_INFO("init beta is " + std::to_string(h_beta[0]));
  }

  LayerNormPlugin(const std::string &name, const void *buffer, size_t length):
    name_(name)
  {
		LOG_INFO("get weight form buffer");
    LOG_INFO("读取epsilon");
    memcpy(&epsilon_, buffer, sizeof(float));

    LOG_INFO("读取length");
    reinterpret_cast<char const *&>(buffer) += sizeof(float);
    memcpy(&length_, buffer, sizeof(std::size_t));
      

    LOG_INFO("读取gammma");
    reinterpret_cast<char const *&>(buffer) += sizeof(std::size_t);
    h_gamma.resize(length_);
    memcpy(h_gamma.data(), buffer, sizeof(float) * length_);

    LOG_INFO("读取beta");
    reinterpret_cast<char const *&>(buffer) += sizeof(float) * length_;
    // float * p2 = static_cast<float *>(malloc(sizeof(float) * 256));
    h_beta.resize(length_);
    memcpy(h_beta.data(), buffer, sizeof(float) * length_);

    reinterpret_cast<char const *&>(buffer) -= sizeof(float) * 256;
    reinterpret_cast<char const *&>(buffer) -= sizeof(std::size_t);
    reinterpret_cast<char const *&>(buffer) -= sizeof(float);
		LOG_INFO("epsilon is " + std::to_string(epsilon_));
		LOG_INFO("length is " + std::to_string(length_));
		LOG_INFO("gamma is " + std::to_string(h_gamma[0]));
		LOG_INFO("beta is " + std::to_string(h_beta[0]));
    WHERE_AM_I();
  }

  LayerNormPlugin() = delete;

  ~LayerNormPlugin()
  {
      WHERE_AM_I();
  }

  size_t getSerializationSize() const noexcept override
  {
      WHERE_AM_I();
      size_t length = sizeof(epsilon_) + sizeof(length_) + sizeof(float) * (
          h_gamma.size() + h_beta.size());
      LOG_INFO("the length of serialized data is " + std::to_string(length));
      return length;
  }

  void serialize(void *buffer) const noexcept override
  {
    WHERE_AM_I();
    LOG_INFO("====== begin serialize ============");

    // 写入epsilon
		LOG_INFO("epsilon is " + std::to_string(epsilon_));
    memcpy(buffer, &epsilon_, sizeof(epsilon_));

    // 写入length
		LOG_INFO("length is " + std::to_string(length_));
    reinterpret_cast<char*&>(buffer) += sizeof(epsilon_);
    memcpy(buffer, &length_, sizeof(length_));

    // 写入gamma
		LOG_INFO("gamma is " + std::to_string(h_gamma[0]));
    reinterpret_cast<char*&>(buffer) += sizeof(length_);
    memcpy(buffer, h_gamma.data(), sizeof(float) * h_gamma.size());

    // 写入beta

		LOG_INFO("beta is " + std::to_string(h_beta[0]));
    reinterpret_cast<char*&>(buffer) += sizeof(float) * h_gamma.size();
    memcpy(buffer, h_beta.data(), sizeof(float) * h_beta.size());

    // 回退buffer指针
    reinterpret_cast<char*&>(buffer) -= sizeof(float) * h_gamma.size();
    reinterpret_cast<char*&>(buffer) -= sizeof(length_);
    reinterpret_cast<char*&>(buffer) -= sizeof(epsilon_);
  }

  IPluginV2DynamicExt *clone() const noexcept override
  {
    WHERE_AM_I();
		LOG_INFO("======== begin clone =========");

		LOG_INFO("epsilon is " + std::to_string(epsilon_));
		LOG_INFO("length is " + std::to_string(length_));
		LOG_INFO("gamma is " + std::to_string(h_gamma[0]));
		LOG_INFO("beta is " + std::to_string(h_beta[0]));
    return new LayerNormPlugin(name_, epsilon_, length_, h_gamma, h_beta);
  }

  int getNbOutputs() const noexcept override
  {
    WHERE_AM_I();
    return 1;
  }

  DimsExprs getOutputDimensions(int32_t outputIndex, const DimsExprs *inputs, int32_t nbInputs, IExprBuilder &exprBuilder) noexcept override
  {
    WHERE_AM_I();
    return inputs[0];
  }

  bool supportsFormatCombination(int32_t pos, const PluginTensorDesc *inOut, int32_t nbInputs, int32_t nbOutputs) noexcept override
  {
    WHERE_AM_I();
    if (inOut[pos].format != TensorFormat::kLINEAR)
    {
        return false;
    }

    bool res = false;
    switch (pos)
    {
    case 0:
        res = (inOut[pos].type == DataType::kFLOAT) || (inOut[pos].type == DataType::kHALF);
        break;
    case 1:
        res = inOut[pos].type == inOut[0].type;
        break;
    default: // should NOT be here
        break;
    }
    return res;
  }

  DataType getOutputDataType(int outputIndex, const DataType *inputTypes, int nbInputs) const noexcept override
  {
    WHERE_AM_I();
    return inputTypes[0];
  }

  void configurePlugin(const DynamicPluginTensorDesc *inputs, int32_t nbInputs, const DynamicPluginTensorDesc *out, int32_t nbOutputs) noexcept override
  {
    WHERE_AM_I();
  }

  size_t getWorkspaceSize(const PluginTensorDesc *inputs, int32_t nbInputs, const PluginTensorDesc *outputs, int32_t nbOutputs) const noexcept override
  {
    WHERE_AM_I();
    return 0;
  }

  void setPluginNamespace(const char *szNamespace) noexcept override
  {
    WHERE_AM_I();
    namespace_ = szNamespace;
  }
  const char *getPluginNamespace() const noexcept override
  {
    WHERE_AM_I();
    return namespace_.c_str();
  }
  const char *getPluginType() const noexcept override
  {
    WHERE_AM_I();
    return PLUGIN_NAME;
  }
  const char *getPluginVersion() const noexcept override
  {
    WHERE_AM_I();
    return PLUGIN_VERSION;
  }
  int initialize() noexcept override
  {
    WHERE_AM_I();
    return 0;
  }
  void terminate() noexcept override
  {
    WHERE_AM_I();
    return;
  }

  void destroy() noexcept override
  {
    // delete this;
    WHERE_AM_I();
  }

  int32_t enqueue(const PluginTensorDesc *inputDesc, const PluginTensorDesc *outputDesc, const void *const *inputs, void *const *outputs, void *workspace, cudaStream_t stream) noexcept override;
}; // class LayerNormPlugin

class LayerNormPluginCreator : public IPluginCreator
{
 private:
  static PluginFieldCollection    fc_;
  static std::vector<PluginField> attr_;
  std::string                     namespace_;

 public:
  LayerNormPluginCreator()
  {
    WHERE_AM_I();
    // 需要收集哪些参数
    attr_.emplace_back(PluginField("epsilon", nullptr, PluginFieldType::kFLOAT32, 1));
    attr_.emplace_back(PluginField("gamma", nullptr, PluginFieldType::kFLOAT32, 1));
    attr_.emplace_back(PluginField("beta", nullptr, PluginFieldType::kFLOAT32, 1));
    fc_.nbFields = attr_.size();
    fc_.fields   = attr_.data();
		LOG_INFO("\nattr_.size=" + std::to_string(attr_.size()));
  }

  ~LayerNormPluginCreator() {}

  IPluginV2 *createPlugin(const char *name, const PluginFieldCollection *fc) noexcept override
  {
    WHERE_AM_I();
    float epsilon {1.0e-5f};
    std::vector<float> gamma;
    std::vector<float> beta;
    std::size_t length = 0;;
    LOG_INFO("============= create plugin  =============");
    LOG_INFO("num of outputs is " + std::to_string(fc->nbFields));
    for (int i = 0; i < fc->nbFields; i++)
    {
        std::string field_name(fc->fields[i].name);
				LOG_INFO("field name is " + field_name);
        if (field_name.compare("epsilon") == 0) {
          epsilon = *static_cast<const float *>(fc->fields[i].data);
        }

        if (field_name.compare("gamma") == 0) {
          const float * temp_gamma = static_cast<const float *>(fc->fields[i].data);
          length = fc->fields[i].length;
          gamma.resize(length);
          std::copy(temp_gamma, temp_gamma + length, gamma.begin());
        }

        if (field_name.compare("beta") == 0) {
          const float * temp_beta = static_cast<const float *>(fc->fields[i].data);
          length = fc->fields[i].length;
          beta.resize(length);
          std::copy(temp_beta, temp_beta + length, beta.begin());
        }
    }

		LOG_INFO("epsilon is " + std::to_string(epsilon));
		LOG_INFO("length is " + std::to_string(length));
		LOG_INFO("gamma is " + std::to_string(gamma[0]));
		LOG_INFO("beta is " + std::to_string(beta[0]));
    LOG_INFO("====================================");
    return new LayerNormPlugin(name, epsilon, length, gamma, beta);
  }

  IPluginV2 *deserializePlugin(const char *name, const void *serialData, size_t serialLength) noexcept override
  {
    WHERE_AM_I();
    return new LayerNormPlugin(name, serialData, serialLength);
  }

  void setPluginNamespace(const char *szNamespace) noexcept override
  {
    WHERE_AM_I();
    namespace_ = szNamespace;
  }

  const char *getPluginNamespace() const noexcept override
  {
    WHERE_AM_I();
    return namespace_.c_str();
  }

  const char *getPluginName() const noexcept override
  {
    WHERE_AM_I();
    return PLUGIN_NAME;
  }

  const char *getPluginVersion() const noexcept override
  {
    WHERE_AM_I();
    return PLUGIN_VERSION;
  }

  const PluginFieldCollection *getFieldNames() noexcept override
  {
    WHERE_AM_I();
    return &fc_;
  }
}; // class LayerNormPluginCreator

} // namespace nvinfer1
