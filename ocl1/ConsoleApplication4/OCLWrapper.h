#include <CL\cl.h>
#include<vector>
class OCLWrapper
{
	public:
		OCLWrapper();
		double* execute(double* a,double* b,int arrSize,const size_t, const size_t);
		void createProgram(const char* kernelSource);
		void setKernel(char* fxnName);
		char* readKernelSource(const char*);
		~OCLWrapper();
	private:
		void initPlatform();
		void initDevice();
		void initContext();
		void initQueue();

		std::vector<double> a, b;
		cl_platform_id platformId;
		cl_device_id deviceId;
		cl_context context;
		cl_command_queue queue;
		cl_program program;
		cl_kernel kern;

};