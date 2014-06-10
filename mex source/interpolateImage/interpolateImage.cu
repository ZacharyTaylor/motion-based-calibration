/* function for projecting lidar points
 *
 */

#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>

#include "../common.h"


__global__ void LinearInterpolateKernel(const float* const imageIn,
										float* const out,
										const size_t height,
										const size_t width,
										const float* const x,
										const float* const y,
										const size_t numPoints){

	unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;

	if(i >= numPoints){
		return;
	}

	if((x[i] < 0) || (y[i] < 0) || (x[i] >= (width-1)) || (y[i] >= (height-1))){
		out[i] = 0;
		return;
	}

	int xF = (int)x[i];
	int yF = (int)y[i];
	float xD = x[i] - (float)xF;
	float yD = y[i] - (float)yF;

	//linear interpolate
	out[i] = (1-yD)*(1-xD)*imageIn[yF + xF*height] +
		(1-yD)*xD*imageIn[yF + (xF+1)*height] +
		yD*(1-xD)*imageIn[yF+1 + xF*height] +
		yD*xD*imageIn[yF+1 + (xF+1)*height];
}

void mexFunction(int nlhs, mxArray *plhs[],
                 int nrhs, mxArray const *prhs[])
{
    //initialize the MathWorks GPU API.
    mxInitGPU();

    //read data
	mxGPUArray const * image = mxGPUCreateFromMxArray(prhs[0]);
    mxGPUArray const * points = mxGPUCreateFromMxArray(prhs[1]);
	size_t imageWidth = mxGPUGetDimensions(image)[1];
	size_t imageHeight = mxGPUGetDimensions(image)[0];
    size_t numPoints = mxGPUGetDimensions(points)[0];
	size_t imageDepth = 1;
	if(mxGPUGetNumberOfDimensions(image) > 2){
		imageDepth = mxGPUGetDimensions(image)[2];
	}
	
    //create pointers from data
    float* imagePtr = (float*)(mxGPUGetDataReadOnly(image));
	float* xPtr = (float*)(mxGPUGetDataReadOnly(points));
    float* yPtr = &(xPtr[numPoints]);

    //create output
    mwSize outSize[] = {numPoints,imageDepth};
    mxGPUArray* out = mxGPUCreateGPUArray(2, outSize, mxSINGLE_CLASS, mxREAL, MX_GPU_DO_NOT_INITIALIZE);
    float* outPtr = (float*)(mxGPUGetDataReadOnly(out));

    //run and get ouputs
	for(size_t i = 0; i < imageDepth; i++){
		float* imageLayerPtr = &(imagePtr[imageHeight*imageWidth*i]);
		float* outLayerPtr =  &(outPtr[numPoints*i]);
		LinearInterpolateKernel<<<gridSize(numPoints), BLOCK_SIZE>>>(imageLayerPtr, outLayerPtr, imageHeight, imageWidth, xPtr, yPtr, numPoints);
		CudaCheckError();
	}
    plhs[0] = mxGPUCreateMxArrayOnGPU(out);
	
    //destroy reference structures
    mxGPUDestroyGPUArray(points);
    mxGPUDestroyGPUArray(image);
    mxGPUDestroyGPUArray(out);
}
