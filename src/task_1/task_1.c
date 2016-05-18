// for docs go here https://github.com/xianyi/OpenBLAS/blob/develop/cblas.h
//C := C + A*B
//test GEMM performance on different BLAST levels

#include <cblas.h>
#include <stdio.h>

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
void GEMM_BLAS1(int leni, int lenk, int lenj, float *A, float *B, float *C) {
  int i, j; 
  for(i = 0; i < leni; i++) {
    for(j = 0; j < lenj; j++) {
        float *x = malloc(sizeof(float)*lenk);
        float *y = malloc(sizeof(float)*lenk);
        memcpy(x, A+i, sizeof(float)*lenk);
        int l;
        for(l=0;l<lenj;l++){
          y[l] = B[j+lenj*l];
        }
        C[lenj*i+j] = cblas_sdot(lenk, x, 1, y, 1);
        //cblas_sdot(OPENBLAS_CONST blasint n, OPENBLAS_CONST float  *x, OPENBLAS_CONST blasint incx, OPENBLAS_CONST float  *y, OPENBLAS_CONST blasint incy);
    }
  }
}

//BLAS-2 level
void GEMM_BLAS2(int leni, int lenk, int lenj, float *A, float *B, float *C) {
  int k; 
  for(k = 0; k < lenk; k++) {
    //cblas_sger (OPENBLAS_CONST enum CBLAS_ORDER order, OPENBLAS_CONST blasint M, OPENBLAS_CONST blasint N, OPENBLAS_CONST float   alpha, OPENBLAS_CONST float  *X, OPENBLAS_CONST blasint incX, OPENBLAS_CONST float  *Y, OPENBLAS_CONST blasint incY, float  *A, OPENBLAS_CONST blasint lda);
  }
}


//BLAS-3 level
void GEMM_BLAS3(int leni, int lenk, int lenj, float *A, float *B, float *C) {
  cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, leni, lenj, lenk, 1, A, 2, B, 2, 1, C, 2);
}

int main() {
  float A[2] = {1,2};
  float B[2] = {3,4};
  float C[4] = {0,0,0,0};
  
  float C_BLAS0[4] = {0,0,0,0};
  float C_BLAS1[4] = {0,0,0,0};
  float C_BLAS2[4] = {0,0,0,0};
  float C_BLAS3[4] = {0,0,0,0};
  
  GEMM_BLAS0(2, 1, 2, A, B, C_BLAS0);
  GEMM_BLAS1(2, 1, 2, A, B, C_BLAS1);
  //GEMM_BLAS2(2, 1, 2, A, B, C_BLAS1);
  GEMM_BLAS3(2, 1, 2, A, B, C_BLAS3);
  
  print_matrix(2, 2, C_BLAS0);
  print_matrix(2, 2, C_BLAS1);
  print_matrix(2, 2, C_BLAS2);
  print_matrix(2, 2, C_BLAS3);
  
  return(0);
}
