## ncu

```
ncu -o profile.1 -k conv2d_naive_kernel --set detailed --import-source yes  python ./run.py
```

When compiling with nvcc, add the flag `-lineinfo`
