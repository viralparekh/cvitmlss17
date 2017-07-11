#!/bin/bash
### warpc ctc installation ##
git clone https://github.com/SeanNaren/warp-ctc.git
cd warp-ctc
mkdir build; cd build
cmake ..
make -j 8
cd ..
export CUDA_HOME="/usr/local/cuda"
cd pytorch_binding
python setup.py install --user


