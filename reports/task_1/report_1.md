# High-Performance Matrix Computations Homework#1

## Implementation

Implemented 4 functions to compute matrix matrix computation on different BLAS levels:

* BLAS-0 (just three nested loops)
* BLAS-1 with cblas_sdot()
* BLAS-2 with cblas_sger()
* BLAS-3 with cblas_sgemm()

All the code is written in C with the use of [OpenBLAS](https://github.com/xianyi/OpenBLAS) library.

## Validation

The validation was done with the use of check_matrix_eq() function.


```c
#define EPS 0.00001

int check_matrix_eq(int leni, int lenj, float *A, float *B) {
  //assume that matrices dimensions are the same
  int i;
  for(i=0; i<leni*lenj;i++){
    if(abs(A[i]-B[i])> EPS){
      return 0;
    }
  }
  return 1;
}
```

Then the solutions of different implementations were checked each with the others.

## Timing

Timing was done like this:

```c
begin = clock();
float *C_BLAS2;
for(i=0;i<ITER;i++){
	C_BLAS2 = calloc(LENI*LENJ, sizeof(float));
	GEMM_BLAS2(LENI, LENJ, LENK, A, B, C_BLAS2);
}
end = clock();
double BLAS2_time = (double)(end-begin) / CLOCKS_PER_SEC;                  
```

For each experiment I keep track of total CPU time, mean and also min and max.

## Performance and Efficiency (A is 1000x1500, B is 1500x1000, 100 iterations)

Funnily enought, profiler gprof does not even count BLAS2 and BLAS3 during the experiments.

```bash
Flat profile:

Each sample counts as 0.01 seconds.
  %   cumulative   self              self     total           
 time   seconds   seconds    calls   s/call   s/call  name    
 63.74    664.86   664.86      100     6.65     6.65  GEMM_BLAS0
 36.47   1045.26   380.40      100     3.80     3.80  GEMM_BLAS1
  0.00   1045.27     0.01      100     0.00     0.00  GEMM_BLAS2
  0.00   1045.27     0.00      100     0.00     0.00  GEMM_BLAS3
  0.00   1045.27     0.00        6     0.00     0.00  check_matrix_eq
  0.00   1045.27     0.00        4     0.00   261.32  experiment
  0.00   1045.27     0.00        2     0.00     0.00  generate_matrix
```

Own output:

```bash
yobibyte@yobibox:~/dev/rwth-hpmc/src/task_1$ ./task_1
#####################
BLAS_0 solution is equal BLAS_1: 1
BLAS_0 solution is equal BLAS_2: 1
BLAS_0 solution is equal BLAS_3: 1
BLAS_1 solution is equal BLAS_2: 1
BLAS_1 solution is equal BLAS_3: 1
BLAS_2 solution is equal BLAS_3: 1
#####################
ITERATIONS DONE: 100
---------------------
BLAS0 CPU total time: 664.337427, mean: 6.643374, min: 6.263151, max: 7.972335
BLAS1 CPU total time: 429.253302, mean: 4.292533, min: 3.986296, max: 4.969887
BLAS2 CPU total time: 63.825768, mean: 0.638258, min: 0.620586, max: 0.708239
BLAS3 CPU total time: 9.702461, mean: 0.097025, min: 0.093930, max: 0.115957
#####################

```

## Architecture details

```bash
yobibyte@yobibox:$ lscpu

Architecture:          x86_64
CPU op-mode(s):        32-bit, 64-bit
Byte Order:            Little Endian
CPU(s):                8
On-line CPU(s) list:   0-7
Thread(s) per core:    2
Core(s) per socket:    4
Socket(s):             1
NUMA node(s):          1
Vendor ID:             GenuineIntel
CPU family:            6
Model:                 60
Model name:            Intel(R) Core(TM) i7-4930MX CPU @ 3.00GHz
Stepping:              3
CPU MHz:               3752.250
CPU max MHz:           3900.0000
CPU min MHz:           800.0000
BogoMIPS:              6385.07
Virtualization:        VT-x
L1d cache:             32K
L1i cache:             32K
L2 cache:              256K
L3 cache:              8192K
NUMA node0 CPU(s):     0-7
```
## Explanation

Memory is expensive in terms of access time. That means: the more we use it (relatively to the number of operations), the worse our performance is. For every memory access operation BLAS3 level uses the most number of computations and that makes it the winner.

I have some doubts about the implementation of BLAS1 and BLAS2 levels. When I want to get the column vector, I do the following:

```c
int l;
for(l=0;l<lenk;l++){
  y[l] = B[j+lenj*l];
}
```

May be there is a more efficient way to do this operation, but I haven't come out with that.

## TODO

Unfortunately, due to the lack of time, the following wasn't done:

* In cache vs. out of cache boundaries
* Performance/Efficiency calculations
* Paying attention to cache usage!
*
