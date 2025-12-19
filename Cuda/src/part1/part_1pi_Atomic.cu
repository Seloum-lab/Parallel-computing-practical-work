/*

This program will numerically compute the integral of

                  4/(1+x*x) 
				  
from 0 to 1.  The value of this integral is pi -- which 
is great since it gives us an easy way to check the answer.

History: Written by Tim Mattson, 11/1999.
         Modified/extended by Jonathan Rouzaud-Cornabas, 10/2022
*/

#include <limits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sys/time.h>

static long num_steps = 100000000;
static long thread_per_block = 32;
static long step_per_thread = 64;

__global__ void computePi(float* sum, long step_per_thread, long num_steps) {
    // Declare and initialize
    extern __device__ __shared__ float sharedLocalSum[];
    float x, localSum = 0.0;
    float step = 1.0/(float) num_steps;
    long threadIndex = threadIdx.x + blockIdx.x*blockDim.x;

    // Initialize the shared variable
    if (threadIdx.x == 0) *sharedLocalSum = 0.0;
    __syncthreads();

    // Compute the local sum per thread
    for (long i = threadIndex * step_per_thread; i < threadIndex * step_per_thread + step_per_thread && i < num_steps; i++) {
        x = (i-0.5)*step;
        localSum += + 4.0/(1.0+x*x);
    }

    // Use two stage atomic add
    atomicAdd(sharedLocalSum, localSum);
    __syncthreads();
    if (threadIdx.x == 0) {
        atomicAdd(sum, *sharedLocalSum);
    }
}

int main (int argc, char** argv)
{
      // Read command line arguments.
      for ( int i = 0; i < argc; i++ ) {
        if ( ( strcmp( argv[ i ], "-N" ) == 0 ) || ( strcmp( argv[ i ], "-num_steps" ) == 0 ) ) {
            num_steps = atol( argv[ ++i ] );
            printf( "  User num_steps is %ld\n", num_steps );
        } else if ( ( strcmp( argv[ i ], "-T" ) == 0 ) || ( strcmp( argv[ i ], "-thread_per_block" ) == 0 ) ) {
            thread_per_block = atol( argv[ ++i ] );
            printf( "  User thread_per_block is %ld\n", thread_per_block );
        } else if ( ( strcmp( argv[ i ], "-S" ) == 0 ) || ( strcmp( argv[ i ], "-step_per_thread" ) == 0 ) ) {
            step_per_thread = atol( argv[ ++i ] );
            printf( "  User step_per_thread is %ld\n", step_per_thread );
        } else if ( ( strcmp( argv[ i ], "-h" ) == 0 ) || ( strcmp( argv[ i ], "-help" ) == 0 ) ) {
            printf( "  Pi Options:\n" );
            printf( "  -num_steps (-N) <int>:      Number of steps to compute Pi (by default 100000000)\n" );
            printf( "  -thread_per_block (-T) <int>:      Number of thread per block (by default 32)\n" );
            printf( "  -step_per_thread (-N) <int>:      Number of steps computed by each thread (by default 64)\n" );
            printf( "  -help (-h):            print this message\n\n" );
            exit( 1 );
        }
      }

      // Kernel launch configuration
      long num_threads = (num_steps / step_per_thread) + (num_steps%step_per_thread ? 1 : 0);
      long num_blocks = (num_threads / thread_per_block) + (num_threads/thread_per_block ? 1 : 0);
      

      // Timer products.
      struct timeval begin, end;
      gettimeofday( &begin, NULL );

      // Initialize
      float pi, sum = 0.0;
      float step = 1.0/(float) num_steps;
      float* d_sum;
      cudaMalloc((void **) &d_sum, sizeof(float));
      cudaMemcpy(d_sum, &sum, sizeof(float), cudaMemcpyHostToDevice);

      // Compute
      computePi<<<num_blocks, thread_per_block, sizeof(float)>>>(d_sum, step_per_thread, num_steps);
      cudaDeviceSynchronize();
      
      // Extract device data
      cudaMemcpy(&sum, d_sum, sizeof(float), cudaMemcpyDeviceToHost);
      cudaFree(d_sum);


      pi = step*sum;

      // Calculate time.
      gettimeofday( &end, NULL );
      double time = 1.0 * ( end.tv_sec - begin.tv_sec ) +
                1.0e-6 * ( end.tv_usec - begin.tv_usec );
                
      printf("\n pi with %ld steps and %ld threads is %f in %lf seconds\n ",num_steps,num_threads,pi,time);
}
