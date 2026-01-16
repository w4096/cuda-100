extern "C" cudaError_t CUDARTAPI __cudaPopCallConfiguration(
  dim3         *gridDim,
  dim3         *blockDim,
  size_t       *sharedMem,
  void         *stream
);

#define __cudaLaunchPrologue(size) \
        void * __args_arr[size]; \
        int __args_idx = 0
        
#define __cudaSetupArg(arg, offset) \
        __args_arr[__args_idx] = (void *)__cudaAddressOf(arg); ++__args_idx
          
#define __cudaSetupArgSimple(arg, offset) \
        __args_arr[__args_idx] = (void *)(char *)&arg; ++__args_idx
        
#if defined(__GNUC__)
#define __NV_ATTR_UNUSED_FOR_LAUNCH __attribute__((unused))
#else  /* !__GNUC__ */
#define __NV_ATTR_UNUSED_FOR_LAUNCH
#endif  /* __GNUC__ */

#ifdef __NV_LEGACY_LAUNCH
/* the use of __args_idx in the expression below avoids host compiler warning about it being an
   unused variable when the launch has no arguments */
#define __cudaLaunch(fun, isTileKernel) \
        { volatile static char *__f __NV_ATTR_UNUSED_FOR_LAUNCH;  __f = fun; \
          dim3 __gridDim, __blockDim;\
          size_t __sharedMem; \
          cudaStream_t __stream; \
          if (__cudaPopCallConfiguration(&__gridDim, &__blockDim, &__sharedMem, &__stream) != cudaSuccess) \
            return; \
          if (isTileKernel) \
            __blockDim.x = __blockDim.y = __blockDim.z = 1; \
          if (__args_idx == 0) {\
            (void)cudaLaunchKernel(fun, __gridDim, __blockDim, &__args_arr[__args_idx], __sharedMem, __stream);\
          } else { \
            (void)cudaLaunchKernel(fun, __gridDim, __blockDim, &__args_arr[0], __sharedMem, __stream);\
          }\
        }
#else  /* !__NV_LEGACY_LAUNCH */
#define __cudaLaunch(fun, isTileKernel) \
        { volatile static char *__f __NV_ATTR_UNUSED_FOR_LAUNCH;  __f = fun; \
          static cudaKernel_t __handle = 0; \
          volatile static bool __tmp __NV_ATTR_UNUSED_FOR_LAUNCH = (__cudaGetKernel(&__handle, (const void *)fun) == cudaSuccess); \
          dim3 __gridDim, __blockDim;\
          size_t __sharedMem; \
          cudaStream_t __stream; \
          if (__cudaPopCallConfiguration(&__gridDim, &__blockDim, &__sharedMem, &__stream) != cudaSuccess) \
            return; \
          if (isTileKernel) \
            __blockDim.x = __blockDim.y = __blockDim.z = 1; \
          if (__args_idx == 0) {\
            (void)__cudaLaunchKernel_helper(__handle, __gridDim, __blockDim, &__args_arr[__args_idx], __sharedMem, __stream);\
          } else { \
            (void)__cudaLaunchKernel_helper(__handle, __gridDim, __blockDim, &__args_arr[0], __sharedMem, __stream);\
          }\
        }
#endif  /* __NV_LEGACY_LAUNCH */

#if defined(__GNUC__)
#define __nv_dummy_param_ref(param) \
        { volatile static void **__ref __attribute__((unused)); __ref = (volatile void **)param; }
#else /* __GNUC__ */
#define __nv_dummy_param_ref(param) \
        { volatile static void **__ref; __ref = (volatile void **)param; }
#endif /* __GNUC__ */

static void ____nv_dummy_param_ref(void *param) __nv_dummy_param_ref(param)





#define __cudaRegisterBinary(X)                                                   \
        __cudaFatCubinHandle = __cudaRegisterFatBinary((void*)&__fatDeviceText); \
        { void (*callback_fp)(void **) =  (void (*)(void **))(X); (*callback_fp)(__cudaFatCubinHandle); __cudaRegisterFatBinaryEnd(__cudaFatCubinHandle); }\
        atexit(__cudaUnregisterBinaryUtil)