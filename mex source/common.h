#ifndef COMMON_H
#define COMMON_H

#include <sstream>

#define PI 3.1415926535897932384626433832795028841971693993751058209749445923078164062

#define MATLAB
#define MATGPU
#define CUDA_ERROR_CHECK

#ifdef MATLAB
	#include <mex.h>
	#define printf mexPrintf
#endif

#ifdef MATGPU
	#include <gpu/mxGPUArray.h>
#endif

#define BLOCK_SIZE 512

#define CudaSafeCall( err ) __cudaSafeCall( err, FILE, __LINE__ )
#define CudaCheckError()    __cudaCheckError( FILE, __LINE__ )

#define FILE (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)

__inline size_t gridSize(size_t num){
	if(num == 0){
		return 1;
	}
	else
	{
		return (size_t)ceil(((float)(num))/((float)(BLOCK_SIZE)));
	}
};

inline void __cudaSafeCall( cudaError err, const char *file, const int line )
{
	#ifdef CUDA_ERROR_CHECK
		if ( cudaSuccess != err )
		{
			printf("CUDA Function Error at %s:%i : %s\n",
			file, line, cudaGetErrorString( err ) );
			cudaDeviceReset();
		}
	#endif

	return;
}

inline void __cudaCheckError( const char *file, const int line )
{
#ifdef CUDA_ERROR_CHECK
	cudaError errCuda = cudaGetLastError();
	if ( cudaSuccess != errCuda ){
		std::ostringstream err; err << "CUDA Kernel Error at " << file << ":" << line << " : " << cudaGetErrorString(errCuda);
		mexErrMsgTxt(err.str().c_str());
		cudaDeviceReset();
	}

	// More careful checking. However, this will affect performance.
	// Comment away if needed.
	errCuda = cudaDeviceSynchronize();
	if( cudaSuccess != errCuda ){
		printf("CUDA Kernel Error with sync failed at %s:%i : %s\n",
		file, line, cudaGetErrorString( errCuda ) );
		cudaDeviceReset();
	}
#endif

return;
}

#endif //COMMON_H
