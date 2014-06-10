MEXSUFFIX  = mexa64
MATLABHOME = /usr/local/MATLAB/R2013b
CUDAHOME   = /usr/local/cuda-6.0
MEX        = g++
NVCC	   = nvcc 

CFLAGS = -fPIC -pthread -DMATLAB_MEX_FILE -ansi -D_GNU_SOURCE -fno-omit-frame-pointer -pthread -O3 -DNDEBUG
NVCCFLAGS  = -O3 -arch sm_30 -use_fast_math -Xcompiler -fpic -cudart=static

LIBS       = -lmx -lmex -lmat -lm -lmwgpu -lcudart
LIBPATH    = -L$(MATLABHOME)/bin/glnxa64 -L$(CUDAHOME)/lib64
INCLUDE    = -I$(MATLABHOME)/extern/include -I$(MATLABHOME)/toolbox/distcomp/gpu/extern/include -Icommon
MEXFLAGS   = -shared -Wl,-rpath-link,$(CUDAHOME)/lib64

PROJECTS = projectLidar projectLidarFlow interpolateImage imageFromLidar projectLidarPan imageFromLidarPan

all: $(PROJECTS)

imageFromLidar: ./imageFromLidar/imageFromLidarBase.o ./imageFromLidar/imageFromLidarLink.o
	$(MEX) -o $@Mex.$(MEXSUFFIX) $^ $() $(MEXFLAGS) $(LIBPATH) $(LIBS) 

projectLidar: ./projectLidar/projectLidarBase.o ./projectLidar/projectLidarLink.o
	$(MEX) -o $@Mex.$(MEXSUFFIX) $^ $() $(MEXFLAGS) $(LIBPATH) $(LIBS) 

imageFromLidarPan: ./imageFromLidarPan/imageFromLidarPanBase.o ./imageFromLidarPan/imageFromLidarPanLink.o
	$(MEX) -o $@Mex.$(MEXSUFFIX) $^ $() $(MEXFLAGS) $(LIBPATH) $(LIBS) 

projectLidarPan: ./projectLidarPan/projectLidarPanBase.o ./projectLidarPan/projectLidarPanLink.o
	$(MEX) -o $@Mex.$(MEXSUFFIX) $^ $() $(MEXFLAGS) $(LIBPATH) $(LIBS) 

projectLidarFlow: ./projectLidarFlow/projectLidarFlowBase.o ./projectLidarFlow/projectLidarFlowLink.o
	$(MEX) -o $@Mex.$(MEXSUFFIX) $^ $() $(MEXFLAGS) $(LIBPATH) $(LIBS) 

interpolateImage: ./interpolateImage/interpolateImageBase.o ./interpolateImage/interpolateImageLink.o
	$(MEX) -o $@Mex.$(MEXSUFFIX) $^ $() $(MEXFLAGS) $(LIBPATH) $(LIBS) 

%Base.o: %.cu
	$(NVCC) $(NVCCFLAGS) $(INCLUDE) -rdc=true -o $@ -dc $<
%Link.o: %Base.o
	$(NVCC) $(NVCCFLAGS) $(INCLUDE) -dlink $< -o $@

clean:
	rm -f *.o
