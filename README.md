###HPMC home assignments

[Course page](http://hpac.rwth-aachen.de/teaching/hpmc-16/)

##Requirements

* [OpenBlas](https://github.com/xianyi/OpenBLAS)

##Assignment 1 GEMM on Different BLAST levels (in progress...)

Goal: Study the performance of different implementations of GEMM

1. Implement at least 4 variants, covering all four BLAS levels.
2. Validate your routines. Make sure you are computing the right quantity.
Explain how the correctness of your code is assessed.
3. Time your routines.
4. Report the Performance and the Efficiency of the variants.
    * Repetitions. Repeat the experiments K times and include statistics (min, max, med, . . . ).
    * Caching. Repeat the experiments, making sure that at each iteration, the matrices are evicted from cache.
5. Present your results (visually).
    * Annotate the in cache vs. out of cache boundaries.
6. Explain the experiments. Comment. And provide your data.
