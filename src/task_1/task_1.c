#include <cblas.h>
#include <stdio.h>

//C := C + A*B
//test GEMM performance on different BLAST levels

void print_matrix(int leni, int lenj, float *matrix) {
  int i, j;
  for(i = 0; i < leni; i++) {
    for(j = 0; j < lenj; j++) {
      printf("%f  ", matrix[lenj*i+j]);
    }
    printf("\n");
  }
  printf("\n");
}

void zeros(int leni, int lenj, float *C) {
  int i, j;
  for(i = 0; i < leni; i++) {
    for(j = 0; j < lenj; j++) {
      C[lenj*i+j] = 0;
    }
  }
}

//BLAS-0 level
void GEMM_BLAS0(int leni, int lenk, int lenj, float *A, float *B, float *C) {
  int i, j, k;
  for(i = 0; i < leni; i++) {
    for(j = 0; j < lenj; j++) {
      for(k = 0; k < lenk; k++) {
        C[lenj*i+j] += A[lenk*i+k]*B[lenj*k+j];
      }
    }
  }
}


//BLAS-1 level

//BLAS-2 level

//BLAS-3 level
void GEMM_BLAS3(int leni, int lenk, int lenj, float *A, float *B, float *C) {
  cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, leni, lenj, lenk, 1, A, 2, B, 2, 1, C, 2);
}

int main() {
  float A[2] = {1,2};
  float B[2] = {3,4};
  float C[4] = {0,0,0,0};
  
  float C_BLAS0[4] = {0,0,0,0};
  float C_BLAS3[4] = {0,0,0,0};
  
  GEMM_BLAS0(2, 1, 2, A, B, C_BLAS0);
  GEMM_BLAS3(2, 1, 2, A, B, C_BLAS3);
  
  print_matrix(2, 2, C_BLAS0);
  print_matrix(2, 2, C_BLAS3);
  
  return(0);
}
