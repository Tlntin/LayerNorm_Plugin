#!/bin/bash
python3 main.py

export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/root/cuda-s/plugin/pyTorch-LayerNorm/plugin
nvcc -g -G \
 -arch=sm_80 \
 -rdc=true \
 /root/cuda-s/plugin/pyTorch-LayerNorm/main1.cpp \
 -std=c++14 \
 -o /root/cuda-s/plugin/pyTorch-LayerNorm/main1 \
 -I /usr/local/cuda/include \
 -L /usr/local/cuda/lib64 \
 -l cudart \
 -l nvinfer \
 -l nvinfer_plugin \
 -l nvonnxparser \
 -L/root/cuda-s/plugin/pyTorch-LayerNorm/plugin  \
 -l layer_norm_plugin \
  && /root/cuda-s/plugin/pyTorch-LayerNorm/main1