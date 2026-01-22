## CUDA

A collection of CUDA code I write when I was learning CUDA programming.

## Get Started

```sh
$ git clone https://github.com/w4096/cuda.git

$ cd cuda
$ git submodule update --init --recursive
```

## Run the Code

I have implemented a Python class which can compile and run the CUDA kernel. In some kernel folder, there is a `run.py` file, you can run this script to test the CUDA kernel.

For example, to run the code in folder `./labs/gemm`, you can use the command:

Set the project root path as `PYTHONPATH` first:

```bash
$ source env.sh  # set project root path as PYTHONPATH
```

then run the script:

```bash
$ cd ./labs/gemm
$ python ./run.py
```

