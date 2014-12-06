#include "OCLWrapper.h"
#include <vector>
#include <iostream>
	
	OCLWrapper::OCLWrapper()
	{
		initPlatform();
		initDevice();
		initContext();
		initQueue();
	}

	void OCLWrapper::initPlatform()
	{
		std::cout << "Platform: " << std::endl;
		cl_uint numPlatforms;
		//get number of platforms
		cl_int errOut = clGetPlatformIDs(0, 0, &numPlatforms);

		//get platform ids
		std::vector<cl_platform_id> platformIds(numPlatforms);
		errOut = clGetPlatformIDs(numPlatforms, &platformIds[0], 0);

		//get platform info

		for (int i = 0; i < numPlatforms; i++)
		{
			//get platform name size
			size_t platformNameLength = 0;
			errOut = clGetPlatformInfo(platformIds[i], CL_PLATFORM_NAME, 0, 0, &platformNameLength);

			//get actual platform name
			std::vector<char> name(platformNameLength);
			errOut = clGetPlatformInfo(platformIds[i], CL_PLATFORM_NAME, platformNameLength, &name[0], 0);
			std::cout << "[" << i << "] " << &name[0] << std::endl;

		}
		std::cout << "Select platform index: ";
		int platIndex;
		std::cin >> platIndex;
		platformId = platformIds[platIndex];
	}

	void OCLWrapper::initDevice()
	{
		std::cout << "Device: " << std::endl;
		cl_uint numDevices=0;
		
		//get number of devices 
		cl_int errOut = clGetDeviceIDs(platformId, CL_DEVICE_TYPE_ALL, 0, 0, &numDevices);

		//get devices ids
		std::vector<cl_device_id> deviceIds(numDevices);
		errOut = clGetDeviceIDs(platformId, CL_DEVICE_TYPE_ALL, numDevices, &deviceIds[0], 0);

		for (int i = 0; i < numDevices; i++)
		{
			//get device name size
			size_t deviceNameLength = 0;
			errOut = clGetDeviceInfo(deviceIds[i], CL_DEVICE_NAME, 0, 0, &deviceNameLength);

			//get actual device name
			std::vector<char> name(deviceNameLength);
			errOut = clGetDeviceInfo(deviceIds[i], CL_DEVICE_NAME, deviceNameLength, &name[0],0);
			std::cout << "[" << i << "] " << &name[0] << std::endl;
		}
		std::cout << "Select device index: ";
		int deviceIndex;
		std::cin >> deviceIndex;
		deviceId = deviceIds[deviceIndex];


	}

	void OCLWrapper::initContext()
	{
		context = clCreateContext(0, 1, &deviceId, 0, 0, 0);
	}

	void OCLWrapper::initQueue()
	{
		cl_int errOut;
		queue= clCreateCommandQueue(context, deviceId, 0, &errOut);
	}

	void OCLWrapper::createProgram(const char* kernelSource)
	{
		cl_int errOut;
		
		program = clCreateProgramWithSource(context, 1,(const char**) &kernelSource, 0,&errOut);
		clBuildProgram(program, 0, NULL, "-g -s .\k1.cl", NULL, NULL);
	}

	void OCLWrapper::setKernel(char* fxnName)
	{
		cl_int errOut;
		kern = clCreateKernel(program, fxnName, &errOut);
	}

	double* OCLWrapper::execute(double* a, double* b, int arrSize, size_t gsize, size_t lsize)
	{
		cl_int errOut;
		cl_mem buffA = clCreateBuffer(context, CL_MEM_READ_ONLY|CL_MEM_ALLOC_HOST_PTR, arrSize*sizeof(double), NULL, &errOut);
		cl_mem buffB = clCreateBuffer(context, CL_MEM_READ_ONLY|CL_MEM_ALLOC_HOST_PTR, arrSize*sizeof(double), NULL, &errOut);
		cl_mem out = clCreateBuffer(context, CL_MEM_WRITE_ONLY|CL_MEM_ALLOC_HOST_PTR, arrSize*sizeof(double), NULL, &errOut);
		cl_double* c=(cl_double*)clEnqueueMapBuffer(queue, out, CL_TRUE, CL_MAP_WRITE, 0, arrSize*sizeof(double), 0, NULL, NULL, &errOut);
		//set buffers as kernel args
		clEnqueueWriteBuffer(queue, buffA, CL_TRUE, 0, arrSize*sizeof(double), a, 0, NULL, NULL);
		clEnqueueWriteBuffer(queue, buffB, CL_TRUE, 0, arrSize*sizeof(double), b, 0, NULL, NULL);

		clSetKernelArg(kern, 0, sizeof(cl_mem), &buffA);
		clSetKernelArg(kern, 1, sizeof(cl_mem), &buffB);

		//output buffer
		clSetKernelArg(kern, 2, sizeof(cl_mem), &out);
		cl_ulong start, end;
		for (int i = 0; i < 3; i++)
		{
			cl_event event;
			clEnqueueNDRangeKernel(queue, kern, 1, NULL, &gsize, &lsize, 0, NULL, &event);
			clFinish(queue);
			clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(start), &start, NULL);
			clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(end), &end, NULL);
			cl_ulong elapsed = end - start;
			std::cout << "took:" << elapsed << "ns" << "| gs:"<<arrSize<<" ls:"<<lsize<< std::endl;
		}
		return c;
	}

	char* OCLWrapper::readKernelSource(const char *filename)
	{
		long int
			size = 0,
			res = 0;

		char *src = NULL;

		FILE *file = fopen(filename, "rb");

		if (!file)  return NULL;

		if (fseek(file, 0, SEEK_END))
		{
			fclose(file);
			return NULL;
		}

		size = ftell(file);
		if (size == 0)
		{
			fclose(file);
			return NULL;
		}

		rewind(file);

		src = (char *)calloc(size + 1, sizeof(char));
		if (!src)
		{
			src = NULL;
			fclose(file);
			return src;
		}

		res = fread(src, 1, sizeof(char) * size, file);
		if (res != sizeof(char) * size)
		{
			fclose(file);
			free(src);

			return src;
		}

		src[size] = '\0'; /* NULL terminated */
		fclose(file);

		return src;
	}

	OCLWrapper::~OCLWrapper()
	{
		clReleaseProgram(program);
		clReleaseKernel(kern);
		clReleaseCommandQueue(queue);
		clReleaseContext(context);
	}