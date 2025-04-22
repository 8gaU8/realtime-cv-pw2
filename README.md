# RealTimeForImageProcessing

## PRACTICAL WORK 2: OPENMP AND CUDA BASED IMAGE PROCESSING

All of exercise codes are available in [`/pw2-omp`](./pw2-omp/) and [`/pw2-cuda`](./pw2-cuda/) directores.

1. Exercise 1 – Image processing methods
    1. Anaglyph
        - OpenMP: [`/pw2-omp/ex1-1-anaglyph.cpp`](./pw2-omp/ex1-1-anaglyph.cpp)
        - CUDA: [`/pw2-cuda/ex2-1-anaglyph.cu`](./pw2-cuda/ex2-1-anaglyph.cu)
    2. Gaussian Filtering
        - OpenMP: [`/pw2-omp/ex1-2-gaussian.cpp`](./pw2-omp/ex1-2-gaussian.cpp)
        - CUDA: [`/pw2-cuda/ex2-2-gaussian.cu`](./pw2-cuda/ex2-2-gaussian.cu)
    3. Denoising
        - OpenMP: [`/pw2-omp/ex1-3-denoising.cpp`](./pw2-omp/ex1-3-denoising.cpp)
        - CUDA: [`/pw2-cuda/ex2-3-denoising.cu`](./pw2-cuda/ex2-3-denoising.cu)

    - Usage
        - OpenMP
            - Build: `cd ./pw2-omp; make all`
            - Execute:
                ```bash
                $ ./ex1-1-anaglyph ../assets/sf.png true 16
                $ ./ex1-2-gaussian ../assets/sf.png true 3 0.5 16
                $ ./ex1-3-denoising../assets/sf.png true 3 10 0.5 16
                ```
        - CUDA 
            - Build: `cd ./pw2-cuda; make all`
            - Execute:
                ```bash
                $ ./ex2-1-anaglyph ../assets/sf.png true 32 8
                $ ./ex2-2-gaussian ../assets/sf.png true 3 0.5 32 8
                $ ./ex2-3-denoising../assets/sf.png true 3 10 0.5 32 8
                ```


2. Exercise 2 – Shared memory optimization

    - **in-progress**
    - file: [`/pw2-cuda/ex2-4-gaussian-shared-memory.cu`](./pw2-cuda/ex2-4-gaussian-shared-memory.cu)


3. Exercise 3 – Execution time comparisons

    - A benchmark script is show in [`/benchmark.py`](./benchmark.py)
    - Results and performance graphs are available in [`/benchmark.xlsx`](./banchmark.xlsx)

