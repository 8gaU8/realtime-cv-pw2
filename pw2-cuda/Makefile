# CUDAコンパイラの設定
NVCC = /usr/local/cuda-12.1/bin/nvcc
CFLAGS = -O3 -diag-suppress=611
OPENCV = `pkg-config opencv4 --cflags --libs`

# ターゲットの定義
TARGETS = ex2-1-anaglyph ex2-2-gaussian ex2-3-denoising ex2-4-gaussian-shared-memory

# デフォルトターゲット
all: $(TARGETS)

ex2-1-anaglyph: ex2-1-anaglyph.cu
	$(NVCC) $(CFLAGS) $< $(OPENCV) -o $@

ex2-2-gaussian: ex2-2-gaussian.cu
	$(NVCC) $(CFLAGS) $< $(OPENCV) -o $@

ex2-3-denoising: ex2-3-denoising.cu
	$(NVCC) $(CFLAGS) $< $(OPENCV) -o $@

ex2-4-gaussian-shared-memory: ex2-4-gaussian-shared-memory.cu
	$(NVCC) $(CFLAGS) $< $(OPENCV) -o $@

# クリーンアップルール
clean:
	rm -f $(TARGETS) *.o
