### 项目说明
- 演示pytorch导出LayerNorm层到onnx文件，然后修改onnx再利用tensorrt进行解析与运行。

### 文件说明
1. plugin：插件目录。
- xx.so为生成的插件, plugin.so与plugin2.so的差别就是前者注释了3行注册插件的代码。具体详情可以参考[链接](https://zhuanlan.zhihu.com/p/524038615)
- cu为cuda-cpp文件，兼容cpp，里面包含两种LayerNorm的算法。
- Makefile，用于生成so文件，对于CUFLAG行，加上-D DEBUG则会显示详情记录，去除则运行so文件时不会显示详情信息。
2. .vscode：用于直接配置Vscode，但如果要通过f5直接运行cpp文件，则仍然需要配置VScode相关插件，具体要装的Vscode插件可以参考[链接](https://www.yuque.com/docs/share/0c853416-a924-497e-9846-b61216416a32?)
3. main.py: 包含pytorch生成onnx代码，以及加载tensorrt插件，并且生成tensort文件并且运行tensorrt的代码。
4. main1.cpp: 用c++的方式来加载插件，生成tensorrt,运行tensorrt文件。
5. main2.py: 另外一种将onnx转tensorrt的方法，方法简洁，适合快速测试插件是否有效。
6. run.sh: 不用vscode，直接命令行运行main.py 与 main1.cpp文件的方法，注意路径是否一致,不一致则自行修改。


### 环境要求
1. 建议安装docker, nvidia-docker2
2. 拉取镜像
```bash
nvidia-docker pull registry.cn-hangzhou.aliyuncs.com/trt2022/dev
```
3. 安装pytorch
4. 安装xtensor: 用于在c++环境下加载numpy生成的npy文件。安装[参考链接](https://xtensor.readthedocs.io/en/latest/installation.html)

### 运行指南

#### python运行
1. 方法1： 运行main.py
2. 方法2：先运行main.py，再运行main2.py 

#### c++运行
1. 方法1：安装好Vscode,配置好Vscode环境，先运行main.py，再用vscode的ctrl+f5运行main1.cpp
2. 方法2：修改好run.sh,直接运行run.sh即可（里面已包含main.py与main1.cpp的运行）。

### 参考链接：
1. [修改onnx相关](https://github.com/NVIDIA/trt-samples-for-hackathon-cn/tree/master/cookbook/08-Tool)
2. [原始版LayerNorm插件](https://github.com/NVIDIA/trt-samples-for-hackathon-cn/tree/master/cookbook/06-PluginAndParser/pyTorch-LayerNorm)
3. [TensorRT_Pro，带tensorrt环境以及全套教程](https://github.com/shouxieai/tensorRT_Pro)

