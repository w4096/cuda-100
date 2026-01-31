## CUDA

This repository contains some CUDA kernels I have implemented when learning CUDA programming.

## Run the Code

If you want to run the code, you can clone this repository and initialize the submodules by running the following commands:

```sh
$ git clone https://github.com/w4096/cuda.git

$ cd cuda
$ git submodule update --init --recursive
$ pip install -r requirements.txt
```

I have implemented a Python script to help compile and run the CUDA kernels easily. In some kernel folders, there are `run.py` scripts, you can run them directly after setting the `PYTHONPATH` environment variable.

For example, to run the code in folder `./labs/gemm`, you can use the command:

```bash
# set PYTHONPATH
$ source env.sh

# then run the script:
$ cd ./labs/gemm
$ python ./run.py
```

All the code are tested on Ubuntu 24.04 with CUDA 13.1 and an NVIDIA RTX 5070 GPU.
