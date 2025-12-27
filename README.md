## CUDA-100

A collection of CUDA code I write when I was learning CUDA programming.

## Run the Code

I have implemented a Python class which can compile and run the CUDA kernel conveniently. In every kernel folder, there is a `run.py` file, you can run this script to test the CUDA kernel.

For example, to run the code in folder `./kernels/relu`, you can use the command:

set the project root path as `PYTHONPATH` first:

```bash
source env.sh  # set project root path as PYTHONPATH
```

then run the script:

```bash
python ./kernels/relu/run.py
```
