# CUDAコンパイラの設定
NVCC = /usr/local/cuda-12.1/bin/nvcc
CFLAGS = -O3 -diag-suppress=611
OPENCV = `pkg-config opencv4 --cflags --libs`

# ターゲットの定義
TARGETS = anaglyph gaussian

# デフォルトターゲット
all: $(TARGETS)

anaglyph: anaglyph.cu
	$(NVCC) $(CFLAGS) $< $(OPENCV) -o $@

gaussian: gaussian.cu
	$(NVCC) $(CFLAGS) $< $(OPENCV) -o $@


# クリーンアップルール
clean:
	rm -f $(TARGETS) *.o
