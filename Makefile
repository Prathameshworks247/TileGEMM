# sm_75 = Tesla T4 (Colab). Override for other GPUs:
#   make ARCH=sm_80   # A100
#   make ARCH=sm_86   # RTX 30xx
NVCC ?= nvcc
ARCH ?= sm_75

matmul: matmul.cu
	$(NVCC) -O3 -arch=$(ARCH) -o $@ $< -lcublas

run: matmul
	./matmul

clean:
	rm -f matmul

.PHONY: run clean
