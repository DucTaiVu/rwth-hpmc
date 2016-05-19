// for docs go here https://github.com/xianyi/OpenBLAS/blob/develop/cblas.h
//C := C + A*B
//test GEMM performance on different BLAST levels

#include <cblas.h>
#include <time.h>
#include <stdio.h>


#define EPS 0.00001
#define ITER 10000

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

int main() {

  //A is leni*lenk matrix
  //B is lenk*lenj matrix
  //C is leni*lenj matirx
  //float A[LENI*LENK] = {1,2};
  //float B[LENK*LENJ] = {3,4};
  float A[LENI*LENK] = {1,2,3,4,5,6};
  float B[LENK*LENJ] = {5,6,7,8,9,10};
  
  int i;  
  clock_t begin, end;

  begin = clock();
  float *C_BLAS0;
  for(i=0;i<ITER;i++){
    C_BLAS0 = (float*)calloc(LENI*LENJ, sizeof(float));
    GEMM_BLAS0(LENI, LENJ, LENK, A, B, C_BLAS0);
  }
  end = clock();
  double BLAS0_time = (double)(end-begin) / CLOCKS_PER_SEC;
  
  begin = clock();
  float *C_BLAS1;
  for(i=0;i<ITER;i++){
    C_BLAS1 = (float*)calloc(LENI*LENJ, sizeof(float));
    GEMM_BLAS1(LENI, LENJ, LENK, A, B, C_BLAS1);
  }
  end = clock();
  double BLAS1_time = (double)(end-begin) / CLOCKS_PER_SEC;
  
  begin = clock();
  float *C_BLAS2;
  for(i=0;i<ITER;i++){
    C_BLAS2 = calloc(LENI*LENJ, sizeof(float));
    GEMM_BLAS2(LENI, LENJ, LENK, A, B, C_BLAS2);
  }
  end = clock();
  double BLAS2_time = (double)(end-begin) / CLOCKS_PER_SEC;
   
  begin = clock();
  float *C_BLAS3;
  for(i=0;i<ITER;i++){
    C_BLAS3 = (float*)calloc(LENI*LENJ, sizeof(float));
    GEMM_BLAS3(LENI, LENJ, LENK, A, B, C_BLAS3);
  }
  end = clock();
  double BLAS3_time = (double)(end-begin) / CLOCKS_PER_SEC;

  //print_matrix(LENI, LENJ, C_BLAS0);
  //print_matrix(LENI, LENJ, C_BLAS1);
  //print_matrix(LENI, LENJ, C_BLAS2);
  //print_matrix(LENI, LENJ, C_BLAS3);
  printf("#####################\n");
  printf("BLAS_0 solution is equal BLAS_1: %d\n", check_matrix_eq(LENI, LENJ, C_BLAS0, C_BLAS1));
  printf("BLAS_0 solution is equal BLAS_2: %d\n", check_matrix_eq(LENI, LENJ, C_BLAS0, C_BLAS2));
  printf("BLAS_0 solution is equal BLAS_3: %d\n", check_matrix_eq(LENI, LENJ, C_BLAS0, C_BLAS3));
  printf("BLAS_1 solution is equal BLAS_2: %d\n", check_matrix_eq(LENI, LENJ, C_BLAS1, C_BLAS2));
  printf("BLAS_1 solution is equal BLAS_3: %d\n", check_matrix_eq(LENI, LENJ, C_BLAS1, C_BLAS3));
  printf("BLAS_2 solution is equal BLAS_3: %d\n", check_matrix_eq(LENI, LENJ, C_BLAS2, C_BLAS3));
  printf("#####################\n");
  printf("ITERATIONS DONE: %d\n", ITER);
  printf("---------------------\n");
  printf("BLAS0 CPU time: %f, per iteration: %f\n", BLAS0_time, BLAS0_time/ITER);
  printf("BLAS1 CPU time: %f, per iteration: %f\n", BLAS1_time, BLAS1_time/ITER);
  printf("BLAS2 CPU time: %f, per iteration: %f\n", BLAS2_time, BLAS2_time/ITER);
  printf("BLAS3 CPU time: %f, per iteration: %f\n", BLAS3_time, BLAS3_time/ITER);
  printf("#####################\n");
 
  return(0);
}

//TODO
//DONE Check all leading dimensios in blas calls.
//DONE check for correctness
//add big matrices for tests
//DONE add timings
//make the report
