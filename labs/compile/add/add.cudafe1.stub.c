#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#pragma GCC diagnostic ignored "-Wcast-qual"
#define __NV_CUBIN_HANDLE_STORAGE__ static
#if !defined(__CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__)
#define __CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__
#endif
#include "crt/host_runtime.h"
#include "add.fatbin.c"
extern __attribute__((visibility("hidden"))) void __device_stub__Z10vector_addPKfS0_Pfi(const float *, const float *, float *, int);
static void __nv_cudaEntityRegisterCallback(void **);
static void __sti____cudaRegisterAll(void) __attribute__((__constructor__));
__attribute__((visibility("hidden"))) void __device_stub__Z10vector_addPKfS0_Pfi(const float *__par0, const float *__par1, float *__par2, int __par3){__cudaLaunchPrologue(4);__cudaSetupArgSimple(__par0, 0UL);__cudaSetupArgSimple(__par1, 8UL);__cudaSetupArgSimple(__par2, 16UL);__cudaSetupArgSimple(__par3, 24UL);__cudaLaunch(((char *)((void ( *)(const float *, const float *, float *, int))vector_add)), 0U);}
# 1 "add.cu"
void vector_add( const float *__cuda_0,const float *__cuda_1,float *__cuda_2,int __cuda_3)
# 1 "add.cu"
{__device_stub__Z10vector_addPKfS0_Pfi( __cuda_0,__cuda_1,__cuda_2,__cuda_3);




}
# 1 "add.cudafe1.stub.c"
static void __nv_cudaEntityRegisterCallback( void **__T0) {  __nv_dummy_param_ref(__T0); __nv_save_fatbinhandle_for_managed_rt(__T0); __cudaRegisterEntry(__T0, ((void ( *)(const float *, const float *, float *, int))vector_add), _Z10vector_addPKfS0_Pfi, (-1)); }
static void __sti____cudaRegisterAll(void) {  __cudaRegisterBinary(__nv_cudaEntityRegisterCallback);  }

#pragma GCC diagnostic pop
