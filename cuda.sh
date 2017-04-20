#!/usr/bin/env bash
# Test script for checking if Cuda and Drivers correctly installed on Ubuntu 14.04, by Roelof Pieters (@graphific)
# BSD License

if [ "$(whoami)" == "root" ]; then
  echo "running as root, please run as user you want to have stuff installed as"
  exit 1
fi
###################################
#   Ubuntu 14.04 Install script for:
# - Nvidia graphic drivers for Titan X: 352
# - Cuda 7.0 (7.5 gives "out of memory" issues)
# - CuDNN3
# - Theano (bleeding edge)
# - Torch7
# - ipython notebook (running as service with circus auto(re)boot on port 8888)
# - itorch notebook (running as service with circus auto(re)boot on port 8889)
# - Caffe
# - OpenCV 3.0 gold release (vs. 2015-06-04)
# - Digits
# - Lasagne
# - Nolearn
# - Keras
###################################

# started with a bare ubuntu 14.04.3 LTS install, with only ubuntu-desktop installed
# script will install the bare minimum, with all "extras" in a seperate venv

export DEBIAN_FRONTEND=noninteractive

# Checking cuda installation
# installing the samples and checking the GPU
cuda-install-samples-7.0.sh ~/
cd NVIDIA\_CUDA-7.0\_Samples/1\_Utilities/deviceQuery
make

#Samples installed and GPU(s) Found ?
./deviceQuery  | grep "Result = PASS"
greprc=$?
if [[ $greprc -eq 0 ]] ; then
    echo "Cuda Samples installed and GPU found"
    echo "you can also check usage and temperature of gpus with nvidia-smi"
else
    if [[ $greprc -eq 1 ]] ; then
        echo "Cuda Samples not installed, exiting..."
        exit 1
    else
        echo "Some sort of error, exiting..."
        exit 1
    fi
fi

echo "now would be time to install cudnn for a speedup"
echo "unfortunately only available by registering on nvidias website:"
echo "https://developer.nvidia.com/cudnn"
echo "deep learning libraries can be installed with final script #3"
