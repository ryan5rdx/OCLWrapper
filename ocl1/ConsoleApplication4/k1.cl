#pragma OPENCL EXTENSION cl_khr_fp64 : enable                   
__kernel void vecAdd(  __global double *a,                      // TODO: Add OpenCL kernel code here.
                       __global double *b,                      
                       __global double *c)                      
{                                                               
    //Get our global thread ID                                  
    int id = get_global_id(0);                                  
    //get itemId within group                                   
	double4 aa=vload4(id,a);								    
	double4 bb=vload4(id,b);									 
	double4 cc=vload4(id,c);                                     
	vstore4(aa*bb,id,c);                                    
}                                                               