// for docs go here https://github.com/xianyi/OpenBLAS/blob/develop/cblas.h
//C := C + A*B
//test GEMM performance on different BLAST levels

#include <cblas.h>
#include <stdio.h>

#define LENI 2
#define LENK 3 
#define LENJ 2


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
void GEMM_BLAS0(int leni, int lenj, int lenk, float *A, float *B, float *C) {
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
void GEMM_BLAS1(int leni, int lenj, int lenk, float *A, float *B, float *C) {
  int i, j; 
  for(i = 0; i < leni; i++) {
    float *x = malloc(sizeof(float)*lenk);
    memcpy(x, A+i*lenk, sizeof(float)*lenk);
    for(j = 0; j < lenj; j++) {
        float *y = malloc(sizeof(float)*lenk);
        int l;
        for(l=0;l<lenk;l++){
          y[l] = B[j+lenj*l];
        }
        C[lenj*i+j] = cblas_sdot(lenk, x, 1, y, 1);
    }
  }
}

void GEMM_BLAS2(int leni, int lenj, int lenk, float *A, float *B, float *C) {
  int l;
  for(l=0; l<lenk; l++){
    float *x = malloc(sizeof(float)*leni);
    float *y = malloc(sizeof(float)*lenj);
    memcpy(y, B+l*lenj, sizeof(float)*lenj);
    int i;
    for(i = 0; i < leni; i++) {
      x[i] = A[l+lenk*i];
    }
    cblas_sger(CblasRowMajor, leni, lenj, 1, x, 1, y, 1, C, lenj);
  }
}

//BLAS-3 level
void GEMM_BLAS3(int leni, int lenj, int lenk, float *A, float *B, float *C) {
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, leni, lenj, lenk, 1.0, A, lenk, B, lenj, 1.0, C, lenj);
}

int main() {

  //A is leni*lenk matrix
  //B is lenk*lenj matrix
  //C is leni*lenj matirx
  //float A[LENI*LENK] = {1,2};
  //float B[LENK*LENJ] = {3,4};
  float A[LENI*LENK] = {1,2,3,4,5,6};
  float B[LENK*LENJ] = {5,6,7,8,9,10};
  
  float *C_BLAS0 = (float*)calloc(LENI*LENJ, sizeof(float));
  float *C_BLAS1 = (float*)calloc(LENI*LENJ, sizeof(float));
  float *C_BLAS2 = calloc(LENI*LENJ, sizeof(float));
  float *C_BLAS3 = (float*)calloc(LENI*LENJ, sizeof(float));

  GEMM_BLAS0(LENI, LENJ, LENK, A, B, C_BLAS0);
  GEMM_BLAS1(LENI, LENJ, LENK, A, B, C_BLAS1);
  GEMM_BLAS2(LENI, LENJ, LENK, A, B, C_BLAS2);
  GEMM_BLAS3(LENI, LENJ, LENK, A, B, C_BLAS3);
  
  print_matrix(LENI, LENJ, C_BLAS0);
  print_matrix(LENI, LENJ, C_BLAS1);
  print_matrix(LENI, LENJ, C_BLAS2);
  print_matrix(LENI, LENJ, C_BLAS3);
 
  return(0);
}

//TODO
//Check all leading dimensios in blas calls.
//add big matrices for tests
//check for correctness
//add timings
//make the report
