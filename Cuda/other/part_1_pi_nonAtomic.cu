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
static long num_threads = 1000;
double step;

__global__ void computePi(double* sum_by_thread, double step, long num_threads, long stepSize, long num_steps) {
    double x, sum = 0.0;
    for (int i = blockIdx.x * stepSize; i < blockIdx.x * stepSize + stepSize && i < num_steps; i++) {
        x = (i-0.5)*step;
        out = sum + 4.0/(1.0+x*x);
    }
    sum_by_thread[blockIdx.x] = sum;
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
            printf( "  User thread_per_block is %ld\n", num_steps );
        } else if ( ( strcmp( argv[ i ], "-S" ) == 0 ) || ( strcmp( argv[ i ], "-step_per_thread" ) == 0 ) ) {
            step_per_thread = atol( argv[ ++i ] );
            printf( "  User step_per_thread is %ld\n", num_steps );
        } else if ( ( strcmp( argv[ i ], "-h" ) == 0 ) || ( strcmp( argv[ i ], "-help" ) == 0 ) ) {
            printf( "  Pi Options:\n" );
            printf( "  -num_steps (-N) <int>:      Number of steps to compute Pi (by default 100000000)\n" );
            printf( "  -thread_per_block (-T) <int>:      Number of steps to compute Pi (by default 100000000)\n" );
            printf( "  -num_steps (-N) <int>:      Number of steps to compute Pi (by default 100000000)\n" );
            printf( "  -help (-h):            print this message\n\n" );
            exit( 1 );
        }
      }
      long stepSize = num_steps / num_threads + 1;
      
	  double pi, sum = 0.0;
	  
      step = 1.0/(double) num_steps;

      // Timer products.
      struct timeval begin, end;
      gettimeofday( &begin, NULL );

      // Initialize
      double* d_localSum;
      int size = num_threads*sizeof(double);
      cudaMalloc((void **) &d_localSum, size);

      // Compute
      computePi<<<num_threads, 1>>>(d_localSum, step, num_threads, stepSize, num_steps);

      // Compute sequentially
      double * localSum = new double[num_threads];
      cudaMemcpy(localSum, d_localSum, size, cudaMemcpyDeviceToHost);
      cudaFree(d_localSum);

      // Compute total sum
      for (int i = 0; i < num_threads; i++) sum += localSum[i];

      pi = step*sum;


      
      gettimeofday( &end, NULL );

      // Calculate time.
      double time = 1.0 * ( end.tv_sec - begin.tv_sec ) +
                1.0e-6 * ( end.tv_usec - begin.tv_usec );
                
      printf("\n pi with %ld steps and %ld threads is %lf in %lf seconds\n ",num_steps,num_threads,pi,time);
}
