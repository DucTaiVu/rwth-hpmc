// for docs go here https://github.com/xianyi/OpenBLAS/blob/develop/cblas.h
//C := C + A*B
//test GEMM performance on different BLAST levels

#include <cblas.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define get_ticks(var) {\
      unsigned int __a, __d;\
      asm volatile("rdtsc" : "=a" (__a), "=d" (__d));\
      var = ((unsigned long) __a) | (((unsigned long) __d) << 32); \
   } while(0)

#define EPS 0.00001
#define ITER 100
#define RANDMAX 10

#define LENI 1000
#define LENK 1500
#define LENJ 1000

struct Timing {
  double total;
  double min;
  double max;
  double mean;
  unsigned long ticks;
};

void print_matrix(int leni, int lenj, double *matrix) {
  int i, j;
  for(i = 0; i < leni; i++) {
    for(j = 0; j < lenj; j++) {
      printf("%f  ", matrix[lenj*i+j]);
    }
    printf("\n");
  }
  printf("\n");
}

//BLAS-0 level
void GEMM_BLAS0(int leni, int lenj, int lenk, double *A, double *B, double *C) {
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
void GEMM_BLAS1(int leni, int lenj, int lenk, double *A, double *B, double *C) {
  int i, j;
  double *x = malloc(sizeof(double)*lenk);
  for(i = 0; i < leni; i++) {
    memcpy(x, A+i*lenk, sizeof(double)*lenk);
    for(j = 0; j < lenj; j++) {
        double *y = malloc(sizeof(double)*lenk);
        int l;
        for(l=0;l<lenk;l++){
          y[l] = B[j+lenj*l];
        }
        C[lenj*i+j] = cblas_ddot(lenk, x, 1, y, 1);
        free(y);
    }
  }
  free(x);
}

void GEMM_BLAS2(int leni, int lenj, int lenk, double *A, double *B, double *C) {
  int l;
  double *x;
  double *y = malloc(sizeof(double)*lenj);

  for(l=0; l<lenk; l++){
    x = malloc(sizeof(double)*leni);
    memcpy(y, B+l*lenj, sizeof(double)*lenj);
    int i;
    for(i = 0; i < leni; i++) {
      x[i] = A[l+lenk*i];
    }
    cblas_dger(CblasRowMajor, leni, lenj, 1, x, 1, y, 1, C, lenj);
  }
  free(x);
  free(y);
}

//BLAS-3 level
void GEMM_BLAS3(int leni, int lenj, int lenk, double *A, double *B, double *C) {
  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, leni, lenj, lenk, 1.0, A, lenk, B, lenj, 1.0, C, lenj);
}


int check_matrix_eq(int leni, int lenj, double *A, double *B) {
  //assume that matrices dimensions are the same
  int i;
  for(i=0; i<leni*lenj;i++){
    if(abs(A[i]-B[i])> EPS){
      return 0;
    }
  }
  return 1;
}

int generate_matrix(int leni, int lenj, double *A) {
  int i;
  for(i=0;i<leni*lenj;i++){
    A[i] = rand()*RANDMAX;
  }
}

struct Timing experiment(int leni, int lenj, int lenk, double *A, double *B, double *C, void (*blas)(int, int, int, double*, double*, double*)){
  int i;
  struct Timing t_s;
  clock_t begin, end, cb, ce;
  unsigned long ticksb, tickse;
  double *attempts_t = malloc(sizeof(double)*ITER);

  begin = clock();
  get_ticks(ticksb);
  for(i=0;i<ITER;i++){
    cb = clock();
    memset(C, 0, sizeof(double)*leni*lenj);
    blas(LENI, LENJ, LENK, A, B, C);
    ce = clock();
    attempts_t[i] = (double)(ce-cb)/CLOCKS_PER_SEC;
  }
  get_ticks(tickse);
  end = clock();
  double time = (double)(end-begin) / CLOCKS_PER_SEC;

  double min_t = INFINITY;
  double max_t = 0;
  for(i=0; i<ITER;i++){
    if (max_t < attempts_t[i]) {
      max_t = attempts_t[i];
    }
    if (min_t > attempts_t[i]) {
      min_t = attempts_t[i];
    }
  }
  t_s.total = time;
  t_s.mean = time/ITER;
  t_s.min = min_t;
  t_s.max = max_t;
  t_s.ticks = tickse-ticksb;
  free(attempts_t);
  return t_s;
}

int main() {

  double *A = (double*)malloc(LENI*LENK*sizeof(double));
  double *B = (double*)malloc(LENK*LENK*sizeof(double));
  generate_matrix(LENI, LENK, A);
  generate_matrix(LENK, LENJ, B);

  double *C_BLAS0 = (double*)malloc(LENI*LENJ*sizeof(double));
  double *C_BLAS1 = (double*)malloc(LENI*LENJ*sizeof(double));
  double *C_BLAS2 = (double*)malloc(LENI*LENJ*sizeof(double));
  double *C_BLAS3 = (double*)malloc(LENI*LENJ*sizeof(double));

  struct Timing t0;
  struct Timing t1;
  struct Timing t2;
  struct Timing t3;

  t0 = experiment(LENI, LENJ, LENK, A, B, C_BLAS0, &GEMM_BLAS0);
  t1 = experiment(LENI, LENJ, LENK, A, B, C_BLAS1, &GEMM_BLAS1);
  t2 = experiment(LENI, LENJ, LENK, A, B, C_BLAS2, &GEMM_BLAS2);
  t3 = experiment(LENI, LENJ, LENK, A, B, C_BLAS3, &GEMM_BLAS3);

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

  //mean, max and min are per iteration
  printf("BLAS0 CPU total time: %f, mean: %f, min: %f, max: %f, %.2f ops/sec\n", t0.total, t0.mean, t0.min, t0.max, t0.ticks/t0.total);
  printf("BLAS1 CPU total time: %f, mean: %f, min: %f, max: %f, %.2f ops/sec\n", t1.total, t1.mean, t1.min, t1.max, t1.ticks/t1.total);
  printf("BLAS2 CPU total time: %f, mean: %f, min: %f, max: %f, %.2f ops/sec\n", t2.total, t2.mean, t2.min, t2.max, t2.ticks/t2.total);
  printf("BLAS3 CPU total time: %f, mean: %f, min: %f, max: %f, %.2f ops/sec\n", t3.total, t3.mean, t3.min, t3.max, t3.ticks/t3.total);
  printf("#####################\n");

  free(A);
  free(B);
  free(C_BLAS0);
  free(C_BLAS1);
  free(C_BLAS2);
  free(C_BLAS3);

  return(0);
}
