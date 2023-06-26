# 컴파일러 및 플래그 설정
CC := nvcc
CFLAGS := -res-usage -use_fast_math -arch=sm_89

# 링크할 라이브러리 및 플래그 설정
LDFLAGS := -lcudart # 필요한 CUDA 라이브러리 추가
LIBS := # 추가적으로 링크할 라이브러리가 있다면 여기에 추가

# 소스 파일 및 실행 파일 이름 설정
SRCS := Genes_NSGA2.cu
EXEC := Genes_NSGA2_exe

# 오브젝트 파일 생성
OBJS := $(SRCS:.cu=.o)

# 빌드 규칙
all: $(EXEC)

$(EXEC): $(OBJS)
	$(CC) $(OBJS) -o $@ $(LDFLAGS) $(LIBS)

# 컴파일 규칙
%.o: %.cu
	$(CC) $(CFLAGS) -c $< -o $@

# 정리 규칙
clean:
	rm -f $(OBJS) $(EXEC)
