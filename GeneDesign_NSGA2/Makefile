CC := nvcc
CFLAGS := -res-usage -use_fast_math --gpu-architecture=sm_89
SRCS := Genes_NSGA2.cu
OBJS := $(SRCS:.cu=.o)
TARGET := Genes_NSGA2_exe
# LDFLAGS := -lcudart
# LIBS := 

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $< -o $@ 

%.o: %.cu
	$(CC) $(CFLAGS) -c $< -o $@

.PHONY: clean
clean:
	rm -f $(OBJS) $(TARGET)
