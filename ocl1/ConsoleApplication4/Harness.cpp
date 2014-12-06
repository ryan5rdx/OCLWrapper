#include <CL\cl.h>
#include "OCLWrapper.h"
#include <iostream>
#include <vector>
#include <string>
int main(int argc, char* argv[])
{
	
	int arrSize = 1000000;
	//total number of kernels that will run(work items)
	int globalSize = arrSize/4;
	//work items per CU work group
	int localSize = 1000;

	OCLWrapper* ocl = new OCLWrapper();
	const char* kernelSource = ocl->readKernelSource("k1.cl");
	ocl->createProgram(kernelSource);
	ocl->setKernel("vecAdd");


	double* a = (double*)(malloc(arrSize*sizeof(double)));
	double* b = (double*)(malloc(arrSize*sizeof(double)));
	for (int i = 0; i <= arrSize; i++)
	{
		a[i]= i;
		b[i]= i;
	}
	double* res=ocl->execute(a,b,arrSize,globalSize,localSize);
	return 0;
	
}

