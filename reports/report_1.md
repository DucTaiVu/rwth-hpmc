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

## Performance and Efficiency
## In cache vs. out of cache boundaries
## Explanation
