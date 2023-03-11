/* include C/C++ header */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

/* include CUDA header */
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <curand_kernel.h>
#include <cuda.h>
#include <cooperative_groups.h>


#define CHECK_CUDA(func)                                                       \
{                                                                              \
    cudaError_t status = (func);                                               \
    if (status != cudaSuccess) {                                               \
        printf("CUDA API failed at line %d with error: %s (%d)\n",             \
               __LINE__, cudaGetErrorString(status), status);                  \
        return EXIT_FAILURE;                                                   \
    }                                                                          \
}       

#define _CRT_SECURE_NO_WARINGS

#define WARP_SIZE 32

#define NOT_FOUND -1
#define CODON_SIZE 3
#define EMPTY -1

#define RANDOM 0
#define UPPER 1
//#define LOWER 2

#define OBJECTIVE_NUM 3
#define _mCAI 0
#define _mHD 1
#define _MLRCS 2

#define P 0
#define Q 1
#define L 2

#define FIRST_SOL 1
#define SECOND_SOL 2

#define IDEAL_MCAI 1
#define IDEAL_MHD 0.4f
#define IDEAL_MLRCS 0
#define EUCLID(val1, val2, val3) (float)sqrt(pow(IDEAL_MCAI - val1, 2) + pow(IDEAL_MHD - val2, 2) + pow(val3, 2))



/* -------------------- 20 kinds of amino acids & weights are sorted ascending order -------------------- */
char Amino_abbreviation[20] = { 'A','C','D','E','F','G','H','I','K','L','M','N','P','Q','R','S','T','V','W','Y' };
char Codons[61 * CODON_SIZE + 1] = "GCGGCAGCCGCU\
UGCUGU\
GACGAU\
GAGGAA\
UUUUUC\
GGGGGAGGCGGU\
CACCAU\
AUAAUCAUU\
AAAAAG\
CUCCUGCUUCUAUUAUUG\
AUG\
AAUAAC\
CCGCCCCCUCCA\
CAGCAA\
CGGCGACGCAGGCGUAGA\
UCGAGCAGUUCAUCCUCU\
ACGACAACCACU\
GUAGUGGUCGUU\
UGG\
UAUUAC";
char Codons_num[20] = { 4,2,2,2,2,4,2,3,2,6,1,2,4,2,6,6,4,4,1,2 };
float Codons_weight[61] = { 1854 / 13563.0f, 5296 / 13563.0f, 7223 / 135063.0f, 1.0f,\
1234 / 3052.0f, 1.0f,\
8960 / 12731.0f, 1.0f,\
6172 / 19532.0f,1.0f,\
7773 / 8251.0f, 1.0f,\
1852 / 15694.0f, 2781 / 15694.0f, 3600 / 15694.0f, 1.0f,\
3288 / 4320.0f, 1.0f,\
3172 / 12071.0f, 8251 / 12071.0f,1.0f,\
12845 / 15169.0f, 1.0f,\
1242 / 13329.0f, 2852 / 13329.0f, 3207 / 13329.0f, 4134 / 13329.0f, 8549 / 13329.0f, 1.0f,\
1.0f,\
8613 / 9875.0f,1.0f,\
1064 / 8965.0f, 1656 / 8965.0f, 4575 / 8965.0f, 1.0f,\
3312 / 10987.0f, 1.0f,\
342 / 9784.0f, 489 / 9784.0f, 658 / 9784.0f, 2175 / 9784.0f,3307 / 9784.0f, 1.0f,\
2112 / 10025.0f, 2623 / 10025.0f, 3873 / 10025.0f, 4583 / 10025.0f, 6403 / 10025.0f, 1.0f,\
1938 / 9812.0f, 5037 / 9812.0f,6660 / 9812.0f, 1.0f,\
3249 / 11442.0f, 3700 / 11442.0f, 6911 / 11442.0f, 1.0f,\
1.0f,\
5768 / 7114.0f, 1.0f };
/* ------------------------------ end of definition ------------------------------ */


/* find index of Amino_abbreviation array matching with input amino abbreviation using binary search */
__host__ int FindAminoIndex(char amino_abbreviation)
{
	int low = 0;
	int high = 20 - 1;
	int mid;

	while (low <= high) {
		mid = (low + high) / 2;

		if (Amino_abbreviation[mid] == amino_abbreviation)
			return mid;
		else if (Amino_abbreviation[mid] > amino_abbreviation)
			high = mid - 1;
		else
			low = mid + 1;
	}

	return NOT_FOUND;
}

/* Minimum distance to optimal objective value(point) */
__host__ float MinEuclid(const float* objval, int pop_size)
{
	float res;
	float tmp;

	res = 100;
	for (int i = 0; i < pop_size; i++) {
		tmp = EUCLID(objval[i * OBJECTIVE_NUM + _mCAI], objval[i * OBJECTIVE_NUM + _mHD], objval[i * OBJECTIVE_NUM + _MLRCS]);
		if (tmp < res)
			res = tmp;
	}

	return res;
}

__constant__ float c_codons_weight[61];
__constant__ char c_amino_startpos[20];
__constant__ char c_codons[61 * CODON_SIZE + 1];
__constant__ char c_codons_num[20];
__constant__ int c_len_amino_seq;
__constant__ int c_cds_num;
__constant__ float c_mprob;

__device__ int lock = 0;


__device__ char FindNum_C(const char* origin, const char* target, const char num_codons)
{
	char i;

	for (i = 0; i < num_codons; i++)
	{
		if (target[0] == origin[i * CODON_SIZE] && target[1] == origin[i * CODON_SIZE + 1] && target[2] == origin[i * CODON_SIZE + 2]) {
			return i;
		}
	}

	return NOT_FOUND;
}

/* mutate codon upper adaptation or randmom adaptation */
__device__ void mutation(curandStateXORWOW* state, const char* codon_info, char* target, char total_num, char origin_pos, const float mprob, const int type)
{
	float cd_prob;
	char new_idx;

	/* 1.0 is included and 0.0 is excluded */
	cd_prob = curand_uniform(state);

	switch (type)
	{
	case RANDOM:
		new_idx = (char)(curand_uniform(state) * total_num);
		if (cd_prob <= mprob && total_num > 1) {
			while (origin_pos == new_idx || new_idx == total_num) {
				new_idx = (char)(curand_uniform(state) * total_num);
			}
			target[0] = codon_info[new_idx * CODON_SIZE];
			target[1] = codon_info[new_idx * CODON_SIZE + 1];
			target[2] = codon_info[new_idx * CODON_SIZE + 2];
		}
		break;

	case UPPER:
		new_idx = (char)(curand_uniform(state) * (total_num - 1 - origin_pos));
		if (cd_prob <= mprob && (origin_pos != (total_num - 1))) {
			while (new_idx == (total_num - 1 - origin_pos)) {
				new_idx = (char)(curand_uniform(state) * (total_num - 1 - origin_pos));
			}
			target[0] = codon_info[(origin_pos + 1 + new_idx) * CODON_SIZE];
			target[1] = codon_info[(origin_pos + 1 + new_idx) * CODON_SIZE + 1];
			target[2] = codon_info[(origin_pos + 1 + new_idx) * CODON_SIZE + 2];
		}
		break;

		/*case LOWER:
			new_idx = (char)(curand_uniform(state) * origin_pos);
			if (cd_prob <= mprob && origin_pos != 0) {
				while (new_idx == origin_pos) {
					new_idx = (char)(curand_uniform(state) * origin_pos);
				}
				target[0] = codon_info[new_idx * CODON_SIZE];
				target[1] = codon_info[new_idx * CODON_SIZE + 1];
				target[2] = codon_info[new_idx * CODON_SIZE + 2];
			}
			break;*/
	}

	return;
}

__device__ bool ParetoComparison(const float* new_objval, const float* old_objval)
{
	// weak pareto dominance
	if ((new_objval[_mCAI] == old_objval[_mCAI]) &&
		(new_objval[_mHD] == old_objval[_mHD]) &&
		(new_objval[_MLRCS] == old_objval[_MLRCS]))
		return false;
	else if ((new_objval[_mCAI] >= old_objval[_mCAI]) &&
		(new_objval[_mHD] >= old_objval[_mHD]) &&
		(new_objval[_MLRCS] <= old_objval[_MLRCS]))
		return true;
	else
		return false;
}

/* curand generator state setting */
__global__ void setup_kernel(curandStateXORWOW* state, int seed)
{
	int id = blockDim.x * blockIdx.x + threadIdx.x;

	/* Each thread gets same seed, a different sequence number, no offset */
	curand_init(seed, id, 0, &state[id]);

	return;
}


/*
Afeter complete GenSolution in global memory
d_sorted_array = 0, 1, .. n - 1
and solution, etc...
*/
__global__ void GenSolution(curandStateXORWOW* state, const char* d_amino_seq_idx, char* d_pop, float* d_objval, char* d_objidx, int* d_lrcsval,int *d_sorted_array)
{
	curandStateXORWOW localState;
	int id;
	char pos;
	int i, j, k, l;
	int idx, seq_idx;
	int num_partition;
	int len_cds, len_sol;
	char lrcs_i, lrcs_j;
	int lrcs_p, lrcs_q, lrcs_l, tmp_l;


	id = blockDim.x * blockIdx.x + threadIdx.x;
	localState = state[id];
	len_cds = c_len_amino_seq * CODON_SIZE;
	len_sol = len_cds * c_cds_num;

	extern __shared__ int smem[];
	__shared__ int* s_lrcs_tid;
	__shared__ int* s_sol_lrcsval;
	__shared__ float* s_sol_objval;
	__shared__ float* s_obj_compute;
	__shared__ char* s_amino_seq_idx;
	__shared__ char* s_sol;
	__shared__ char* s_sol_objidx;

	s_lrcs_tid = smem;
	s_sol_lrcsval = (int*)&s_lrcs_tid[blockDim.x];
	s_sol_objval = (float*)&s_sol_lrcsval[3];
	s_obj_compute = (float*)&s_sol_objval[OBJECTIVE_NUM];
	s_amino_seq_idx = (char*)&s_obj_compute[blockDim.x];
	s_sol = (char*)&s_amino_seq_idx[c_len_amino_seq];
	s_sol_objidx = (char*)&s_sol[len_sol];


	num_partition = (c_len_amino_seq % blockDim.x == 0) ? c_len_amino_seq / blockDim.x : c_len_amino_seq / blockDim.x + 1;
	for (i = 0; i < num_partition; i++) {
		idx = blockDim.x * i + threadIdx.x;
		if (idx < c_len_amino_seq) {
			s_amino_seq_idx[idx] = d_amino_seq_idx[idx];
		}
	}
	__syncthreads();
	// --------------------------------------------------------------------------------------


	/* initialize solution */
	if (blockIdx.x == gridDim.x - 1)
	{
		num_partition = ((c_len_amino_seq * c_cds_num) % blockDim.x == 0) ? (c_len_amino_seq * c_cds_num) / blockDim.x : (c_len_amino_seq * c_cds_num) / blockDim.x + 1;
		for (i = 0; i < num_partition; i++) {
			idx = blockDim.x * i + threadIdx.x;
			if (idx < c_len_amino_seq * c_cds_num) {
				seq_idx = idx % c_len_amino_seq;

				pos = c_codons_num[s_amino_seq_idx[seq_idx]] - 1;

				j = idx * CODON_SIZE;
				k = (c_amino_startpos[s_amino_seq_idx[seq_idx]] + pos) * CODON_SIZE;

				s_sol[j] = c_codons[k];
				s_sol[j + 1] = c_codons[k + 1];
				s_sol[j + 2] = c_codons[k + 2];
			}
		}
	}
	else {
		num_partition = ((c_len_amino_seq * c_cds_num) % blockDim.x == 0) ? (c_len_amino_seq * c_cds_num) / blockDim.x : (c_len_amino_seq * c_cds_num) / blockDim.x + 1;
		for (i = 0; i < num_partition; i++) {
			idx = blockDim.x * i + threadIdx.x;
			if (idx < c_len_amino_seq * c_cds_num) {
				seq_idx = idx % c_len_amino_seq;

				do {
					pos = (char)(curand_uniform(&localState) * c_codons_num[s_amino_seq_idx[seq_idx]]);
				} while (pos == c_codons_num[s_amino_seq_idx[seq_idx]]);

				j = idx * CODON_SIZE;
				k = (c_amino_startpos[s_amino_seq_idx[seq_idx]] + pos) * CODON_SIZE;

				s_sol[j] = c_codons[k];
				s_sol[j + 1] = c_codons[k + 1];
				s_sol[j + 2] = c_codons[k + 2];
			}
		}
	}
	__syncthreads();

	/* calculate mCAI */
	num_partition = (c_len_amino_seq % blockDim.x == 0) ? (c_len_amino_seq / blockDim.x) : (c_len_amino_seq / blockDim.x) + 1;
	for (i = 0; i < c_cds_num; i++) {
		s_obj_compute[threadIdx.x] = 1;

		for (j = 0; j < num_partition; j++) {
			seq_idx = blockDim.x * j + threadIdx.x;
			if (seq_idx < c_len_amino_seq) {
				pos = FindNum_C(&c_codons[c_amino_startpos[s_amino_seq_idx[seq_idx]] * CODON_SIZE], &s_sol[len_cds * i + seq_idx * CODON_SIZE],
					c_codons_num[s_amino_seq_idx[seq_idx]]);
				s_obj_compute[threadIdx.x] *= (float)pow(c_codons_weight[c_amino_startpos[s_amino_seq_idx[seq_idx]] + pos], 1.0 / c_len_amino_seq);
			}
		}
		__syncthreads();

		j = blockDim.x / 2;
		while (j != 0) {
			if (threadIdx.x < j) {
				s_obj_compute[threadIdx.x] *= s_obj_compute[threadIdx.x + j];
			}
			__syncthreads();

			j /= 2;
		}

		if (threadIdx.x == 0) {
			if (i == 0) {
				s_sol_objval[_mCAI] = s_obj_compute[0];
				s_sol_objidx[_mCAI * 2] = i;
			}
			else if (s_obj_compute[0] <= s_sol_objval[_mCAI]) {
				s_sol_objval[_mCAI] = s_obj_compute[0];
				s_sol_objidx[_mCAI * 2] = i;
			}
		}
		__syncthreads();

	}


	/* calculate mHD */
	num_partition = (len_cds % blockDim.x == 0) ? (len_cds / blockDim.x) : (len_cds / blockDim.x) + 1;
	for (i = 0; i < c_cds_num; i++) {
		for (j = i + 1; j < c_cds_num; j++) {
			s_obj_compute[threadIdx.x] = 0;

			for (k = 0; k < num_partition; k++) {
				seq_idx = blockDim.x * k + threadIdx.x;

				if (seq_idx < len_cds && (s_sol[len_cds * i + seq_idx] != s_sol[len_cds * j + seq_idx])) {
					s_obj_compute[threadIdx.x] += 1;
				}
			}
			__syncthreads();

			k = blockDim.x / 2;
			while (k != 0) {
				if (threadIdx.x < k) {
					s_obj_compute[threadIdx.x] += s_obj_compute[threadIdx.x + k];
				}
				__syncthreads();

				k /= 2;
			}

			if (threadIdx.x == 0) {
				if (i == 0 && j == 1) {
					s_sol_objval[_mHD] = s_obj_compute[0] / len_cds;
					s_sol_objidx[_mHD * 2] = i;
					s_sol_objidx[_mHD * 2 + 1] = j;
				}
				else if ((s_obj_compute[0] / len_cds) <= s_sol_objval[_mHD]) {
					s_sol_objval[_mHD] = s_obj_compute[0] / len_cds;
					s_sol_objidx[_mHD * 2] = i;
					s_sol_objidx[_mHD * 2 + 1] = j;
				}
			}
			__syncthreads();

		}
	}

	/* calculate MLRCS */
	s_obj_compute[threadIdx.x] = NOT_FOUND;
	lrcs_l = 0;
	for (i = 0; i < c_cds_num; i++) {
		for (j = i; j < c_cds_num; j++) {
			idx = threadIdx.x;

			if (i == j)
			{
				while (idx < 2 * len_cds + 1)
				{
					if (idx < len_cds + 1) {
						l = idx + 1;
						seq_idx = len_cds - l;

						for (k = 0; k < l; k++) {
							if (k == 0 || (seq_idx == -1))
								tmp_l = 0;
							else if (s_sol[len_cds * i + seq_idx + k] == s_sol[len_cds * j + k - 1]) {
								tmp_l++;
								if (tmp_l >= lrcs_l) {
									lrcs_l = tmp_l;
									s_obj_compute[threadIdx.x] = lrcs_l;
									lrcs_p = seq_idx + k + 1 - lrcs_l;
									lrcs_q = k - lrcs_l;
									lrcs_i = (char)i;
									lrcs_j = (char)j;
								}
							}
							else
								tmp_l = 0;
						}
					}
					else {
						l = 2 * len_cds + 1 - idx;
						seq_idx = len_cds - l;

						for (k = 0; k < l; k++) {
							if (k == 0)
								tmp_l = 0;
							else if (s_sol[len_cds * i + k - 1] == s_sol[len_cds * j + seq_idx + k])
							{
								tmp_l++;
								if (tmp_l >= lrcs_l) {
									lrcs_l = tmp_l;
									s_obj_compute[threadIdx.x] = lrcs_l;
									lrcs_p = k - lrcs_l;
									lrcs_q = seq_idx + k + 1 - lrcs_l;
									lrcs_i = (char)i;
									lrcs_j = (char)j;
								}
							}
							else
								tmp_l = 0;
						}

					}

					idx += blockDim.x;
				}
			}
			else
			{
				while (idx < 2 * len_cds + 1)
				{
					if (idx < len_cds + 1) {
						l = idx + 1;
						seq_idx = len_cds - l;
						for (k = 0; k < l; k++) {
							if (k == 0)
								tmp_l = 0;
							else if (s_sol[len_cds * i + seq_idx + k] == s_sol[len_cds * j + k - 1]) {
								tmp_l++;
								if (tmp_l >= lrcs_l) {
									lrcs_l = tmp_l;
									s_obj_compute[threadIdx.x] = lrcs_l;
									lrcs_p = seq_idx + k + 1 - lrcs_l;
									lrcs_q = k - lrcs_l;
									lrcs_i = (char)i;
									lrcs_j = (char)j;
								}
							}
							else
								tmp_l = 0;
						}
					}
					else {
						l = 2 * len_cds + 1 - idx;
						seq_idx = len_cds - l;

						for (k = 0; k < l; k++) {
							if (k == 0)
								tmp_l = 0;
							else if (s_sol[len_cds * i + k - 1] == s_sol[len_cds * j + seq_idx + k])
							{
								tmp_l++;
								if (tmp_l >= lrcs_l) {
									lrcs_l = tmp_l;
									s_obj_compute[threadIdx.x] = lrcs_l;
									lrcs_p = k - lrcs_l;
									lrcs_q = seq_idx + k + 1 - lrcs_l;
									lrcs_i = (char)i;
									lrcs_j = (char)j;
								}
							}
							else
								tmp_l = 0;
						}

					}

					idx += blockDim.x;
				}
			}

		}
	}
	__syncthreads();

	j = blockDim.x / 2;
	s_lrcs_tid[threadIdx.x] = threadIdx.x;
	__syncthreads();
	while (j != 0)
	{
		if (threadIdx.x < j && (s_obj_compute[threadIdx.x + j] > s_obj_compute[threadIdx.x]))
		{
			s_obj_compute[threadIdx.x] = s_obj_compute[threadIdx.x + j];
			s_lrcs_tid[threadIdx.x] = s_lrcs_tid[threadIdx.x + j];
		}
		__syncthreads();

		j /= 2;
	}

	if (threadIdx.x == s_lrcs_tid[0])
	{
		s_sol_lrcsval[L] = lrcs_l;
		s_sol_lrcsval[P] = lrcs_p;
		s_sol_lrcsval[Q] = lrcs_q;

		s_sol_objval[_MLRCS] = (float)lrcs_l / len_cds;
		s_sol_objidx[_MLRCS * 2] = lrcs_i;
		s_sol_objidx[_MLRCS * 2 + 1] = lrcs_j;
	}
	__syncthreads();

	/* copy from shared memory to global memory */
	num_partition = (len_sol % blockDim.x == 0) ? (len_sol / blockDim.x) : (len_sol / blockDim.x) + 1;
	for (i = 0; i < num_partition; i++) {
		idx = blockDim.x * i + threadIdx.x;
		if (idx < len_sol)
			d_pop[blockIdx.x * len_sol + idx] = s_sol[idx];
	}

	if (threadIdx.x == 0)
	{
		d_objval[blockIdx.x * OBJECTIVE_NUM + _mCAI] = s_sol_objval[_mCAI];
		d_objval[blockIdx.x * OBJECTIVE_NUM + _mHD] = s_sol_objval[_mHD];
		d_objval[blockIdx.x * OBJECTIVE_NUM + _MLRCS] = s_sol_objval[_MLRCS];

		d_objidx[blockIdx.x * OBJECTIVE_NUM * 2 + _mCAI * 2] = s_sol_objidx[_mCAI * 2];
		d_objidx[blockIdx.x * OBJECTIVE_NUM * 2 + _mHD * 2] = s_sol_objidx[_mHD * 2];
		d_objidx[blockIdx.x * OBJECTIVE_NUM * 2 + _mHD * 2 + 1] = s_sol_objidx[_mHD * 2 + 1];
		d_objidx[blockIdx.x * OBJECTIVE_NUM * 2 + _MLRCS * 2] = s_sol_objidx[_MLRCS * 2];
		d_objidx[blockIdx.x * OBJECTIVE_NUM * 2 + _MLRCS * 2 + 1] = s_sol_objidx[_MLRCS * 2 + 1];

		d_lrcsval[blockIdx.x * 3 + P] = s_sol_lrcsval[P];
		d_lrcsval[blockIdx.x * 3 + Q] = s_sol_lrcsval[Q];
		d_lrcsval[blockIdx.x * 3 + L] = s_sol_lrcsval[L];
		
		/* write d_sorted_array */
		d_sorted_array[blockIdx.x] = blockIdx.x;
	}

	state[id] = localState;

	return;
}


/* 
Based on sorting methods on NSGA2 paper
If we input number of threads and sorting function into cudaOccupancyMaxActiveBlocksPerMultiprocessor(), we get number of blocks available to use
*/
__global__ void FastSortSolution(int* d_sorted_array, const float* d_objval, int* F_set, int* Sp_set, int * Q_set, int* np, const int pop_size)
{
	cooperative_groups::grid_group grid = cooperative_groups::this_grid();
	grid.sync();

	int i, j;
	int p;
	int Sp_idx;
	float tmp;
	float crw_dist;
	int p_len;

	
	extern __shared__ int smem[];
	__shared__ int* F_idx;
	__shared__ int* F_front;
	__shared__ int* Q_idx;
	__shared__ float* obj_compute;

	np = smem;
	Q_set = (int*)&np[pop_size];
	F_idx = (int*)&Q_set[pop_size];
	F_front = (int*)&F_idx[1];
	Q_idx = (int*)&F_front[1];
	obj_compute = (float*)Q_idx[1];

	// Setting empty
	for (i = 0; i < pop_size; i++) {
		F_set[pop_size * threadIdx.x + i] = EMPTY;
		Sp_set[pop_size * threadIdx.x + i] = EMPTY;
	}
	np[threadIdx.x] = 0;
	
	// 1st front setting
	if (threadIdx.x == 0) 
		*F_idx = 0;
	__syncthreads();
	Sp_idx = 0;
	for (i = 0; i < pop_size; i++)
	{
		if (i != threadIdx.x) 
		{
			if (ParetoComparison(&d_objval[threadIdx.x * OBJECTIVE_NUM], &d_objval[i * OBJECTIVE_NUM]))
			{
				Sp_set[threadIdx.x * pop_size + Sp_idx++] = i;
			}
			else if (ParetoComparison(&d_objval[i * OBJECTIVE_NUM], &d_objval[threadIdx.x * OBJECTIVE_NUM]))
			{
				np[threadIdx.x] += 1;
			}
		}
	}
	__syncthreads();
	if (np[threadIdx.x] == 0)
	{
		while (atomicCAS(&lock, 0, 1) != 0);
		F_set[*F_idx] = threadIdx.x;
		*F_idx += 1;
		atomicExch(&lock, 0);
	}
	__syncthreads();
	// end of 1st front setting


	// non-dominated sorting
	if (threadIdx.x == 0) {
		*F_front = 0;
	}
	__syncthreads();
	while (F_set[*F_front * pop_size] != EMPTY)
	{
		Q_set[threadIdx.x] = EMPTY;
		if (threadIdx.x == 0)
			*Q_idx = 0;
		__syncthreads();

		if (F_set[*F_front * pop_size + threadIdx.x] != EMPTY)
		{
			p = F_set[*F_front * pop_size + threadIdx.x];
			for (Sp_idx = 0; Sp_idx < pop_size && Sp_set[p * pop_size + Sp_idx] != EMPTY; Sp_idx++)
			{
				while (atomicCAS(&lock, 0, 1) != 0);
				np[Sp_set[p * pop_size + Sp_idx]] -= 1;
				if (np[Sp_set[p * pop_size + Sp_idx]] == 0)
				{
					// rank ++
					Q_set[*Q_idx] = Sp_set[p * pop_size + Sp_idx];
					*Q_idx += 1;
				}
				atomicExch(&lock, 0);
			}
		}
		__syncthreads();

		if (threadIdx.x == 0)
			*F_front += 1;
		__syncthreads();
		if (*F_front == pop_size)
			break;
		if (threadIdx.x == 0)
		{
			*Q_idx = 0;
			while (*Q_idx < pop_size && Q_set[*Q_idx] != EMPTY)
			{
				F_set[*F_front * pop_size + *Q_idx] = Q_set[*Q_idx];
				*Q_idx += 1;
			}
		}
		__syncthreads();
		if (*Q_idx >= pop_size)
			break;
	}
	// end of non-dominated sorting


	// crowding distance 
	if (threadIdx.x == 0)
		*F_front = 0;
	__syncthreads();
	while (F_set[*F_front * pop_size] != EMPTY)
	{

		if (F_set[*F_front * pop_size + threadIdx.x] != EMPTY)
		{
			crw_dist = 0;
			for (i = 0; i < OBJECTIVE_NUM; i++) {
				obj_compute[threadIdx.x] = d_objval[F_set[*F_front * pop_size + threadIdx.x] * OBJECTIVE_NUM + i];
			}
		}
		__syncthreads();

	}


	// write d_sorted_array
	if (threadIdx.x == 0)
	{
		*F_front = 0;
		i = 0;
		j = 0;
		
		while (F_set[*F_front * pop_size] != EMPTY)
		{
			d_sorted_array[j] = F_set[*F_front * pop_size + i];
			j++;
			i++;
			
			if (j == pop_size)
				break;
			
			if (F_set[*F_front * pop_size + i] == EMPTY) {
				*F_front += 1;
				i = 0;
			}
		}

	}

	return;
}


/*
copy solution from global memory based on sorted array
Not update solution to global memory state of sorted Just write solution to global memory
This means In global memory solution is not sorted after this function
If you want soltion is sorted sorted function call and we get sorted array and update solution based on sorted array
*/
__global__ void mainKernel(curandStateXORWOW* state, const char* d_amino_seq_idx, char* d_pop, float* d_objval, char* d_objidx, int* d_lrcsval, const int cycle, const int *d_sorted_array)
{
	curandStateXORWOW localState;
	int id;
	char pos;
	int i, j, k, l;
	int idx, seq_idx;
	int num_partition;
	int len_cds, len_sol;
	char lrcs_i, lrcs_j;
	int lrcs_p, lrcs_q, lrcs_l, tmp_l;
	char sol_num;

	char* ptr_origin_sol, * ptr_target_sol;
	float* ptr_origin_objval, * ptr_target_objval;
	char* ptr_origin_objidx, * ptr_target_objidx;
	int* ptr_origin_lrcsval, * ptr_target_lrcsval;					// P, Q, L


	id = blockDim.x * blockIdx.x + threadIdx.x;
	localState = state[id];
	len_cds = c_len_amino_seq * CODON_SIZE;
	len_sol = len_cds * c_cds_num;


	/* -------------------- shared memory allocation -------------------- */
	extern __shared__ int smem[];
	__shared__ char* s_amino_seq_idx;
	__shared__ char* s_sol1;
	__shared__ char* s_sol2;
	__shared__ char* s_sol1_objidx;
	__shared__ char* s_sol2_objidx;
	__shared__ char* mutation_type;
	__shared__ float* s_obj_compute;										// for computing mCAI & mHD value
	__shared__ float* s_sol1_objval;
	__shared__ float* s_sol2_objval;
	__shared__ int* s_sol1_lrcsval;
	__shared__ int* s_sol2_lrcsval;
	__shared__ int* s_lrcs_tid;

	s_lrcs_tid = smem;
	s_sol1_lrcsval = (int*)&s_lrcs_tid[blockDim.x];							// for finding which thread have LRCS
	s_sol2_lrcsval = (int*)&s_sol1_lrcsval[3];
	s_obj_compute = (float*)&s_sol2_lrcsval[3];
	s_sol1_objval = (float*)&s_obj_compute[blockDim.x];
	s_sol2_objval = (float*)&s_sol1_objval[OBJECTIVE_NUM];
	s_amino_seq_idx = (char*)&s_sol2_objval[OBJECTIVE_NUM];
	s_sol1 = (char*)&s_amino_seq_idx[c_len_amino_seq];
	s_sol2 = (char*)&s_sol1[len_sol];
	s_sol1_objidx = (char*)&s_sol2[len_sol];
	s_sol2_objidx = (char*)&s_sol1_objidx[OBJECTIVE_NUM * 2];
	mutation_type = (char*)&s_sol2_objidx[OBJECTIVE_NUM * 2];
	/* -------------------- end of shared memory allocation -------------------- */


	sol_num = FIRST_SOL;
	ptr_origin_sol = s_sol1;
	ptr_origin_objval = s_sol1_objval;
	ptr_origin_objidx = s_sol1_objidx;
	ptr_origin_lrcsval = s_sol1_lrcsval;
	ptr_target_sol = s_sol2;
	ptr_target_objval = s_sol2_objval;
	ptr_target_objidx = s_sol2_objidx;
	ptr_target_lrcsval = s_sol2_lrcsval;


	num_partition = (c_len_amino_seq % blockDim.x == 0) ? c_len_amino_seq / blockDim.x : c_len_amino_seq / blockDim.x + 1;
	for (i = 0; i < num_partition; i++) {
		idx = blockDim.x * i + threadIdx.x;
		if (idx < c_len_amino_seq) {
			s_amino_seq_idx[idx] = d_amino_seq_idx[idx];
		}
	}
	// -----------------------------------------------------------------------------------------


	/* copy solution from global memory to shared memory based on sorted array index */
	num_partition = (len_sol % blockDim.x == 0) ? (len_sol / blockDim.x) : (len_sol / blockDim.x) + 1;
	for (i = 0; i < num_partition; i++) {
		idx = blockDim.x * i + threadIdx.x;
		if (idx < len_sol)
			ptr_origin_sol[idx] = d_pop[d_sorted_array[blockIdx.x] * len_sol + idx];
	}

	if (threadIdx.x == 0)
	{
		ptr_origin_objval[_mCAI] = d_objval[d_sorted_array[blockIdx.x] * OBJECTIVE_NUM + _mCAI];
		ptr_origin_objval[_mHD] = d_objval[d_sorted_array[blockIdx.x] * OBJECTIVE_NUM + _mHD];
		ptr_origin_objval[_MLRCS] = d_objval[d_sorted_array[blockIdx.x] * OBJECTIVE_NUM + _MLRCS];

		ptr_origin_objidx[_mCAI * 2] = d_objidx[d_sorted_array[blockIdx.x] * OBJECTIVE_NUM * 2 + _mCAI * 2];
		ptr_origin_objidx[_mHD * 2] = d_objidx[d_sorted_array[blockIdx.x] * OBJECTIVE_NUM * 2 + _mHD * 2];
		ptr_origin_objidx[_mHD * 2 + 1] = d_objidx[d_sorted_array[blockIdx.x] * OBJECTIVE_NUM * 2 + _mHD * 2 + 1];
		ptr_origin_objidx[_MLRCS * 2] = d_objidx[d_sorted_array[blockIdx.x] * OBJECTIVE_NUM * 2 + _MLRCS * 2];
		ptr_origin_objidx[_MLRCS * 2 + 1] = d_objidx[d_sorted_array[blockIdx.x] * OBJECTIVE_NUM * 2 + _MLRCS * 2 + 1];

		ptr_origin_lrcsval[P] = d_lrcsval[d_sorted_array[blockIdx.x] * 3 + P];
		ptr_origin_lrcsval[Q] = d_lrcsval[d_sorted_array[blockIdx.x] * 3 + Q];
		ptr_origin_lrcsval[L] = d_lrcsval[d_sorted_array[blockIdx.x] * 3 + L];
	}
	__syncthreads();



	/* mutate cycle times */
	for (int c = 0; c < cycle; c++)
	{
		/* copy from original solution to target solution */
		num_partition = (len_sol % blockDim.x == 0) ? (len_sol / blockDim.x) : (len_sol / blockDim.x) + 1;
		for (i = 0; i < num_partition; i++)
		{
			seq_idx = blockDim.x * i + threadIdx.x;
			if (seq_idx < len_sol)
			{
				ptr_target_sol[seq_idx] = ptr_origin_sol[seq_idx];
			}
		}

		/* select mutatation type */
		if (threadIdx.x == 0) {
			do {
				*mutation_type = (char)(curand_uniform(&localState) * 4);
			} while (*mutation_type == 4);
		}
		__syncthreads();


		switch (*mutation_type)
		{
		case 0:			// all random
			num_partition = ((c_len_amino_seq * c_cds_num) % blockDim.x == 0) ? (c_len_amino_seq * c_cds_num) / blockDim.x : (c_len_amino_seq * c_cds_num) / blockDim.x + 1;
			for (i = 0; i < num_partition; i++) {
				idx = blockDim.x * i + threadIdx.x;
				if (idx < c_len_amino_seq * c_cds_num) {
					seq_idx = idx % c_len_amino_seq;

					pos = FindNum_C(&c_codons[c_amino_startpos[s_amino_seq_idx[seq_idx]] * CODON_SIZE], &ptr_target_sol[idx * CODON_SIZE],
						c_codons_num[s_amino_seq_idx[seq_idx]]);
					mutation(&localState, &c_codons[c_amino_startpos[s_amino_seq_idx[seq_idx]] * CODON_SIZE], &ptr_target_sol[idx * CODON_SIZE],
						c_codons_num[s_amino_seq_idx[seq_idx]], pos, c_mprob, RANDOM);
				}
			}
			break;

		case 1:			// mCAI
			num_partition = (c_len_amino_seq % blockDim.x == 0) ? (c_len_amino_seq / blockDim.x) : (c_len_amino_seq / blockDim.x) + 1;
			for (i = 0; i < num_partition; i++) {
				seq_idx = blockDim.x * i + threadIdx.x;
				if (seq_idx < c_len_amino_seq) {
					pos = FindNum_C(&c_codons[c_amino_startpos[s_amino_seq_idx[seq_idx]] * CODON_SIZE],
						&ptr_target_sol[len_cds * ptr_origin_objidx[_mCAI * 2] + seq_idx * CODON_SIZE], c_codons_num[s_amino_seq_idx[seq_idx]]);
					mutation(&localState, &c_codons[c_amino_startpos[s_amino_seq_idx[seq_idx]] * CODON_SIZE],
						&ptr_target_sol[len_cds * ptr_origin_objidx[_mCAI * 2] + seq_idx * CODON_SIZE], c_codons_num[s_amino_seq_idx[seq_idx]], pos, c_mprob, UPPER);
				}
			}
			break;

		case 2:			// mHD
			num_partition = (c_len_amino_seq % blockDim.x == 0) ? (c_len_amino_seq / blockDim.x) : (c_len_amino_seq / blockDim.x) + 1;
			for (i = 0; i < num_partition; i++) {
				seq_idx = blockDim.x * i + threadIdx.x;
				if (seq_idx < c_len_amino_seq) {
					pos = FindNum_C(&c_codons[c_amino_startpos[s_amino_seq_idx[seq_idx]] * CODON_SIZE],
						&ptr_target_sol[len_cds * ptr_origin_objidx[_mHD * 2] + seq_idx * CODON_SIZE], c_codons_num[s_amino_seq_idx[seq_idx]]);
					mutation(&localState, &c_codons[c_amino_startpos[s_amino_seq_idx[seq_idx]] * CODON_SIZE],
						&ptr_target_sol[len_cds * ptr_origin_objidx[_mHD * 2] + seq_idx * CODON_SIZE], c_codons_num[s_amino_seq_idx[seq_idx]], pos, c_mprob, RANDOM);

					pos = FindNum_C(&c_codons[c_amino_startpos[s_amino_seq_idx[seq_idx]] * CODON_SIZE],
						&ptr_target_sol[len_cds * ptr_origin_objidx[_mHD * 2 + 1] + seq_idx * CODON_SIZE], c_codons_num[s_amino_seq_idx[seq_idx]]);
					mutation(&localState, &c_codons[c_amino_startpos[s_amino_seq_idx[seq_idx]] * CODON_SIZE],
						&ptr_target_sol[len_cds * ptr_origin_objidx[_mHD * 2 + 1] + seq_idx * CODON_SIZE], c_codons_num[s_amino_seq_idx[seq_idx]], pos, c_mprob, RANDOM);

				}
			}
			break;

		case 3:
			seq_idx = ptr_origin_lrcsval[P] / CODON_SIZE + threadIdx.x;
			while (seq_idx <= (ptr_origin_lrcsval[P] + ptr_origin_lrcsval[L] - 1) / CODON_SIZE)
			{
				pos = FindNum_C(&c_codons[c_amino_startpos[s_amino_seq_idx[seq_idx]] * CODON_SIZE],
					&ptr_target_sol[len_cds * ptr_origin_objidx[_MLRCS * 2] + seq_idx * CODON_SIZE], c_codons_num[s_amino_seq_idx[seq_idx]]);
				mutation(&localState, &c_codons[c_amino_startpos[s_amino_seq_idx[seq_idx]] * CODON_SIZE],
					&ptr_target_sol[len_cds * ptr_origin_objidx[_MLRCS * 2] + seq_idx * CODON_SIZE], c_codons_num[s_amino_seq_idx[seq_idx]], pos, c_mprob, RANDOM);

				seq_idx += blockDim.x;
			}

			seq_idx = ptr_origin_lrcsval[Q] / CODON_SIZE + threadIdx.x;
			while (seq_idx <= (ptr_origin_lrcsval[Q] + ptr_origin_lrcsval[L] - 1) / CODON_SIZE)
			{
				pos = FindNum_C(&c_codons[c_amino_startpos[s_amino_seq_idx[seq_idx]] * CODON_SIZE],
					&ptr_target_sol[len_cds * ptr_origin_objidx[_MLRCS * 2 + 1] + seq_idx * CODON_SIZE], c_codons_num[s_amino_seq_idx[seq_idx]]);
				mutation(&localState, &c_codons[c_amino_startpos[s_amino_seq_idx[seq_idx]] * CODON_SIZE],
					&ptr_target_sol[len_cds * ptr_origin_objidx[_MLRCS * 2 + 1] + seq_idx * CODON_SIZE], c_codons_num[s_amino_seq_idx[seq_idx]], pos, c_mprob, RANDOM);

				seq_idx += blockDim.x;
			}

			break;
		}
		__syncthreads();


		/* calculate mCAI */
		num_partition = (c_len_amino_seq % blockDim.x == 0) ? (c_len_amino_seq / blockDim.x) : (c_len_amino_seq / blockDim.x) + 1;
		for (i = 0; i < c_cds_num; i++) {
			s_obj_compute[threadIdx.x] = 1;

			for (j = 0; j < num_partition; j++) {
				seq_idx = blockDim.x * j + threadIdx.x;
				if (seq_idx < c_len_amino_seq) {
					pos = FindNum_C(&c_codons[c_amino_startpos[s_amino_seq_idx[seq_idx]] * CODON_SIZE], &ptr_target_sol[len_cds * i + seq_idx * CODON_SIZE],
						c_codons_num[s_amino_seq_idx[seq_idx]]);
					s_obj_compute[threadIdx.x] *= (float)pow(c_codons_weight[c_amino_startpos[s_amino_seq_idx[seq_idx]] + pos], 1.0 / c_len_amino_seq);
				}
			}
			__syncthreads();

			j = blockDim.x / 2;
			while (j != 0) {
				if (threadIdx.x < j) {
					s_obj_compute[threadIdx.x] *= s_obj_compute[threadIdx.x + j];
				}
				__syncthreads();

				j /= 2;
			}

			if (threadIdx.x == 0) {
				if (i == 0) {
					ptr_target_objval[_mCAI] = s_obj_compute[0];
					ptr_target_objidx[_mCAI * 2] = i;
				}
				else if (s_obj_compute[0] <= ptr_target_objval[_mCAI]) {
					ptr_target_objval[_mCAI] = s_obj_compute[0];
					ptr_target_objidx[_mCAI * 2] = i;
				}
			}
			__syncthreads();

		}

		/* calculate mHD */
		num_partition = (len_cds % blockDim.x == 0) ? (len_cds / blockDim.x) : (len_cds / blockDim.x) + 1;
		for (i = 0; i < c_cds_num; i++) {
			for (j = i + 1; j < c_cds_num; j++) {
				s_obj_compute[threadIdx.x] = 0;

				for (k = 0; k < num_partition; k++) {
					seq_idx = blockDim.x * k + threadIdx.x;

					if (seq_idx < len_cds && (ptr_target_sol[len_cds * i + seq_idx] != ptr_target_sol[len_cds * j + seq_idx])) {
						s_obj_compute[threadIdx.x] += 1;
					}
				}
				__syncthreads();

				k = blockDim.x / 2;
				while (k != 0) {
					if (threadIdx.x < k) {
						s_obj_compute[threadIdx.x] += s_obj_compute[threadIdx.x + k];
					}
					__syncthreads();

					k /= 2;
				}

				if (threadIdx.x == 0) {
					if (i == 0 && j == 1) {
						ptr_target_objval[_mHD] = s_obj_compute[0] / len_cds;
						ptr_target_objidx[_mHD * 2] = i;
						ptr_target_objidx[_mHD * 2 + 1] = j;
					}
					else if (s_obj_compute[0] / len_cds <= ptr_target_objval[_mHD]) {
						ptr_target_objval[_mHD] = s_obj_compute[0] / len_cds;
						ptr_target_objidx[_mHD * 2] = i;
						ptr_target_objidx[_mHD * 2 + 1] = j;
					}
				}
				__syncthreads();

			}
		}

		/* calculate MLRCS */
		s_obj_compute[threadIdx.x] = NOT_FOUND;
		lrcs_l = 0;
		for (i = 0; i < c_cds_num; i++) {
			for (j = i; j < c_cds_num; j++) {
				idx = threadIdx.x;

				if (i == j)
				{
					while (idx < 2 * len_cds + 1)
					{
						if (idx < len_cds + 1) {
							l = idx + 1;
							seq_idx = len_cds - l;

							for (k = 0; k < l; k++) {
								if (k == 0 || (seq_idx == -1))
									tmp_l = 0;
								else if (ptr_target_sol[len_cds * i + seq_idx + k] == ptr_target_sol[len_cds * j + k - 1]) {
									tmp_l++;
									if (tmp_l >= lrcs_l) {
										lrcs_l = tmp_l;
										s_obj_compute[threadIdx.x] = lrcs_l;
										lrcs_p = seq_idx + k + 1 - lrcs_l;
										lrcs_q = k - lrcs_l;
										lrcs_i = (char)i;
										lrcs_j = (char)j;
									}
								}
								else
									tmp_l = 0;
							}
						}
						else {
							l = 2 * len_cds + 1 - idx;
							seq_idx = len_cds - l;

							for (k = 0; k < l; k++) {
								if (k == 0)
									tmp_l = 0;
								else if (ptr_target_sol[len_cds * i + k - 1] == ptr_target_sol[len_cds * j + seq_idx + k])
								{
									tmp_l++;
									if (tmp_l >= lrcs_l) {
										lrcs_l = tmp_l;
										s_obj_compute[threadIdx.x] = lrcs_l;
										lrcs_p = k - lrcs_l;
										lrcs_q = seq_idx + k + 1 - lrcs_l;
										lrcs_i = (char)i;
										lrcs_j = (char)j;
									}
								}
								else
									tmp_l = 0;
							}

						}

						idx += blockDim.x;
					}
				}
				else
				{
					while (idx < 2 * len_cds + 1)
					{
						if (idx < len_cds + 1) {
							l = idx + 1;
							seq_idx = len_cds - l;
							for (k = 0; k < l; k++) {
								if (k == 0)
									tmp_l = 0;
								else if (ptr_target_sol[len_cds * i + seq_idx + k] == ptr_target_sol[len_cds * j + k - 1]) {
									tmp_l++;
									if (tmp_l >= lrcs_l) {
										lrcs_l = tmp_l;
										s_obj_compute[threadIdx.x] = lrcs_l;
										lrcs_p = seq_idx + k + 1 - lrcs_l;
										lrcs_q = k - lrcs_l;
										lrcs_i = (char)i;
										lrcs_j = (char)j;
									}
								}
								else
									tmp_l = 0;
							}
						}
						else {
							l = 2 * len_cds + 1 - idx;
							seq_idx = len_cds - l;

							for (k = 0; k < l; k++) {
								if (k == 0)
									tmp_l = 0;
								else if (ptr_target_sol[len_cds * i + k - 1] == ptr_target_sol[len_cds * j + seq_idx + k])
								{
									tmp_l++;
									if (tmp_l >= lrcs_l) {
										lrcs_l = tmp_l;
										s_obj_compute[threadIdx.x] = lrcs_l;
										lrcs_p = k - lrcs_l;
										lrcs_q = seq_idx + k + 1 - lrcs_l;
										lrcs_i = (char)i;
										lrcs_j = (char)j;
									}
								}
								else
									tmp_l = 0;
							}

						}

						idx += blockDim.x;
					}
				}

			}
		}
		__syncthreads();

		j = blockDim.x / 2;
		s_lrcs_tid[threadIdx.x] = threadIdx.x;
		__syncthreads();
		while (j != 0)
		{
			if (threadIdx.x < j && s_obj_compute[threadIdx.x + j] > s_obj_compute[threadIdx.x])
			{
				s_obj_compute[threadIdx.x] = s_obj_compute[threadIdx.x + j];
				s_lrcs_tid[threadIdx.x] = s_lrcs_tid[threadIdx.x + j];
			}
			__syncthreads();

			j /= 2;
		}

		if (threadIdx.x == s_lrcs_tid[0])
		{
			ptr_target_lrcsval[L] = lrcs_l;
			ptr_target_lrcsval[P] = lrcs_p;
			ptr_target_lrcsval[Q] = lrcs_q;

			ptr_target_objval[_MLRCS] = (float)lrcs_l / len_cds;
			ptr_target_objidx[_MLRCS * 2] = lrcs_i;
			ptr_target_objidx[_MLRCS * 2 + 1] = lrcs_j;
		}
		__syncthreads();


		if (ParetoComparison(ptr_target_objval, ptr_origin_objval))
		{
			if (sol_num == FIRST_SOL)
				sol_num = SECOND_SOL;
			else
				sol_num = FIRST_SOL;
		}

		if (sol_num == FIRST_SOL) {
			ptr_origin_sol = s_sol1;
			ptr_origin_objval = s_sol1_objval;
			ptr_origin_objidx = s_sol1_objidx;
			ptr_origin_lrcsval = s_sol1_lrcsval;
			ptr_target_sol = s_sol2;
			ptr_target_objval = s_sol2_objval;
			ptr_target_objidx = s_sol2_objidx;
			ptr_target_lrcsval = s_sol2_lrcsval;
		}
		else {
			ptr_origin_sol = s_sol2;
			ptr_origin_objval = s_sol2_objval;
			ptr_origin_objidx = s_sol2_objidx;
			ptr_origin_lrcsval = s_sol2_lrcsval;
			ptr_target_sol = s_sol1;
			ptr_target_objval = s_sol1_objval;
			ptr_target_objidx = s_sol1_objidx;
			ptr_target_lrcsval = s_sol1_lrcsval;
		}

	}


	/* copy from shared memory to global memory */
	num_partition = (len_sol % blockDim.x == 0) ? (len_sol / blockDim.x) : (len_sol / blockDim.x) + 1;
	for (i = 0; i < num_partition; i++) {
		idx = blockDim.x * i + threadIdx.x;
		if (idx < len_sol) {
			d_pop[blockIdx.x * len_sol + idx] = ptr_origin_sol[idx];
			d_pop[(gridDim.x + blockIdx.x) * len_sol + idx] = ptr_target_sol[idx];
		}
	}

	if (threadIdx.x == 0)
	{
		d_objval[blockIdx.x * OBJECTIVE_NUM + _mCAI] = ptr_origin_objval[_mCAI];
		d_objval[blockIdx.x * OBJECTIVE_NUM + _mHD] = ptr_origin_objval[_mHD];
		d_objval[blockIdx.x * OBJECTIVE_NUM + _MLRCS] = ptr_origin_objval[_MLRCS];
		d_objval[(gridDim.x + blockIdx.x) * OBJECTIVE_NUM + _mCAI] = ptr_target_objval[_mCAI];
		d_objval[(gridDim.x + blockIdx.x) * OBJECTIVE_NUM + _mHD] = ptr_target_objval[_mHD];
		d_objval[(gridDim.x + blockIdx.x) * OBJECTIVE_NUM + _MLRCS] = ptr_target_objval[_MLRCS];

		d_objidx[blockIdx.x * OBJECTIVE_NUM * 2 + _mCAI * 2] = ptr_origin_objidx[_mCAI * 2];
		d_objidx[blockIdx.x * OBJECTIVE_NUM * 2 + _mHD * 2] = ptr_origin_objidx[_mHD * 2];
		d_objidx[blockIdx.x * OBJECTIVE_NUM * 2 + _mHD * 2 + 1] = ptr_origin_objidx[_mHD * 2 + 1];
		d_objidx[blockIdx.x * OBJECTIVE_NUM * 2 + _MLRCS * 2] = ptr_origin_objidx[_MLRCS * 2];
		d_objidx[blockIdx.x * OBJECTIVE_NUM * 2 + _MLRCS * 2 + 1] = ptr_origin_objidx[_MLRCS * 2 + 1];
		d_objidx[(gridDim.x + blockIdx.x) * OBJECTIVE_NUM * 2 + _mCAI * 2] = ptr_target_objidx[_mCAI * 2];
		d_objidx[(gridDim.x + blockIdx.x) * OBJECTIVE_NUM * 2 + _mHD * 2] = ptr_target_objidx[_mHD * 2];
		d_objidx[(gridDim.x + blockIdx.x) * OBJECTIVE_NUM * 2 + _mHD * 2 + 1] = ptr_target_objidx[_mHD * 2 + 1];
		d_objidx[(gridDim.x + blockIdx.x) * OBJECTIVE_NUM * 2 + _MLRCS * 2] = ptr_target_objidx[_MLRCS * 2];
		d_objidx[(gridDim.x + blockIdx.x) * OBJECTIVE_NUM * 2 + _MLRCS * 2 + 1] = ptr_target_objidx[_MLRCS * 2 + 1];

		d_lrcsval[blockIdx.x * 3 + P] = ptr_origin_lrcsval[P];
		d_lrcsval[blockIdx.x * 3 + Q] = ptr_origin_lrcsval[Q];
		d_lrcsval[blockIdx.x * 3 + L] = ptr_origin_lrcsval[L];
		d_lrcsval[(gridDim.x + blockIdx.x) * 3 + P] = ptr_target_lrcsval[P];
		d_lrcsval[(gridDim.x + blockIdx.x) * 3 + Q] = ptr_target_lrcsval[Q];
		d_lrcsval[(gridDim.x + blockIdx.x) * 3 + L] = ptr_target_lrcsval[L];
	}

	state[id] = localState;						// update state of random number generator


	return;
}



int main()
{
	srand((unsigned int)time(NULL));

	/* To get information of Deivce */
	int dev = 0;							// number of device (GPU)
	int maxSharedMemPerBlock;
	int maxSharedMemPerProcessor;
	int totalConstantMem;
	int maxRegisterPerProcessor;
	int maxRegisterPerBlock;
	int totalMultiProcessor;
	cudaDeviceProp deviceProp;

	CHECK_CUDA(cudaGetDeviceProperties(&deviceProp, dev))
	CHECK_CUDA(cudaDeviceGetAttribute(&maxSharedMemPerBlock, cudaDevAttrMaxSharedMemoryPerBlock, dev))
	CHECK_CUDA(cudaDeviceGetAttribute(&maxSharedMemPerProcessor, cudaDevAttrMaxRegistersPerMultiprocessor, dev))
	CHECK_CUDA(cudaDeviceGetAttribute(&totalConstantMem, cudaDevAttrTotalConstantMemory, dev))
	CHECK_CUDA(cudaDeviceGetAttribute(&maxRegisterPerProcessor, cudaDevAttrMaxRegistersPerMultiprocessor, dev))
	CHECK_CUDA(cudaDeviceGetAttribute(&maxRegisterPerBlock, cudaDevAttrMaxRegistersPerBlock, dev))
	CHECK_CUDA(cudaDeviceGetAttribute(&totalMultiProcessor, cudaDevAttrMultiProcessorCount, dev))

	printf("Device #%d:\n", dev);
	printf("Name: %s\n", deviceProp.name);
	printf("Compute capability: %d.%d\n", deviceProp.major, deviceProp.minor);
	printf("Clock rate: %d MHz\n", deviceProp.clockRate / 1000);
	printf("Global memory size: %lu MB\n", deviceProp.totalGlobalMem / (1024 * 1024));
	printf("Max thread dimensions: (%d, %d, %d)\n", deviceProp.maxThreadsDim[0], deviceProp.maxThreadsDim[1], deviceProp.maxThreadsDim[2]);
	printf("Max grid dimensions: (%d, %d, %d)\n", deviceProp.maxGridSize[0], deviceProp.maxGridSize[1], deviceProp.maxGridSize[2]);
	printf("Total constant memory: %d bytes\n", totalConstantMem);
	printf("Max threads per SM: %d\n", deviceProp.maxThreadsPerMultiProcessor);
	printf("Max threads per block: %d\n", deviceProp.maxThreadsPerBlock);
	printf("Maximum shared memory per SM: %d bytes\n", maxSharedMemPerProcessor);
	printf("Maximum shared memory per block: %d bytes\n", maxSharedMemPerBlock);
	printf("Maximum number of registers per SM: %d\n", maxRegisterPerProcessor);
	printf("Maximum number of registers per block: %d\n", maxRegisterPerBlock);
	printf("Total number of SM in device: %d\n", totalMultiProcessor);
	printf("\n");

	char input_file[32];
	char* amino_seq;						// store amino sequences from input file
	char* h_amino_seq_idx;					// notify index of amino abbreviation array corresponding input amino sequences
	char* h_amino_startpos;					// notify position of according amino abbreviation index
	char* h_pop;							// store population (a set of solutions)
	float* h_objval;						// store objective values of population (solution 1, solution 2 .... solution n)
	char* h_objidx;
	int* h_lrcsval;
	int* h_sorted_array;					// for check sorting in GPU
	int len_amino_seq, len_cds, len_sol;
	int pop_size;
	int total_cycle;
	int sorting_cycle;
	int cds_num;							// size of solution equal to number of CDSs(codon sequences) in a solution
	float mprob;							// mutation probability
	float min_dist;

	char tmp;
	int i, j, k;
	int x;
	int idx;
	char buf[256];
	FILE* fp;

	int numBlocks;
	int threadsPerBlock;

	char* d_amino_seq_idx;
	char* d_pop;
	float* d_objval;
	char* d_objidx;
	int* d_lrcsval;
	int* d_sorted_array;
	int* d_F_set, * d_Sp_set, *d_Q_set, *d_np;
	curandStateXORWOW* genState;

	cudaEvent_t d_start, d_end;
	float kernel_time;
	CHECK_CUDA(cudaEventCreate(&d_start))
	CHECK_CUDA(cudaEventCreate(&d_end))


	/* input parameter values */
	printf("input file name : "); scanf("%s", input_file);
	printf("input number of cycle : "); scanf("%d", &total_cycle);
	if (total_cycle < 0) {
		printf("input max cycle value >= 0\n");
		return EXIT_FAILURE;
	}
	printf("input number of sorting cycle : "); scanf("%d", &sorting_cycle);
	if (sorting_cycle <= 0) {
		printf("input sorting cycle value > 0\n");
		return EXIT_FAILURE;
	}
	printf("input number of solution : "); scanf("%d", &pop_size);
	if (pop_size <= 0) {
		printf("input number of solution > 0\n");
		return EXIT_FAILURE;
	}
	printf("input number of CDSs in a solution : "); scanf("%d", &cds_num);
	if (cds_num <= 1) {
		printf("input number of CDSs > 1\n");
		return EXIT_FAILURE;
	}
	printf("input mutation probability (0 ~ 1 value) : "); scanf("%f", &mprob);
	if (mprob < 0 || mprob > 1) {
		printf("input mutation probability (0 ~ 1 value) : \n");
		return EXIT_FAILURE;
	}
	printf("input thread per block x value --> number of thread  warp size (32) * x : "); scanf("%d", &x);


	/* read input file (fasta format) */
	fp = fopen(input_file, "r");
	if (fp == NULL) {
		printf("Line : %d Opening input file is failed", __LINE__);
		return EXIT_FAILURE;
	}

	fseek(fp, 0, SEEK_END);
	len_amino_seq = ftell(fp);
	fseek(fp, 0, SEEK_SET);
	fgets(buf, 256, fp);
	len_amino_seq -= ftell(fp);

	amino_seq = (char*)malloc(sizeof(char) * len_amino_seq);

	idx = 0;
	while (!feof(fp)) {
		tmp = fgetc(fp);
		if (tmp != '\n')
			amino_seq[idx++] = tmp;
	}
	amino_seq[idx] = NULL;
	len_amino_seq = idx - 1;
	len_cds = len_amino_seq * CODON_SIZE;
	len_sol = len_cds * cds_num;

	fclose(fp);
	/* end file process */

	h_amino_seq_idx = (char*)malloc(sizeof(char) * len_amino_seq);
	for (i = 0; i < len_amino_seq; i++) {
		idx = FindAminoIndex(amino_seq[i]);
		if (idx == NOT_FOUND) {
			printf("FindAminoIndex function is failed... \n");
			return EXIT_FAILURE;
		}
		h_amino_seq_idx[i] = idx;
	}

	h_amino_startpos = (char*)malloc(sizeof(char) * 20);
	h_amino_startpos[0] = 0;
	for (i = 1; i < 20; i++) {
		h_amino_startpos[i] = h_amino_startpos[i - 1] + Codons_num[i - 1];
	}

	numBlocks = pop_size;
	threadsPerBlock = WARP_SIZE * x;

	/* host memory allocation */
	h_pop = (char*)malloc(sizeof(char) * pop_size * len_sol * 2);
	h_objval = (float*)malloc(sizeof(float) * pop_size * OBJECTIVE_NUM * 2);
	h_objidx = (char*)malloc(sizeof(char) * pop_size * OBJECTIVE_NUM * 2 * 2);
	h_lrcsval = (int*)malloc(sizeof(int) * pop_size * 3 * 2);
	h_sorted_array = (int*)malloc(sizeof(int) * pop_size * 2);


	/* device memory allocation */
	CHECK_CUDA(cudaMalloc((void**)&genState, sizeof(curandStateXORWOW) * numBlocks * threadsPerBlock))
	CHECK_CUDA(cudaMalloc((void**)&d_amino_seq_idx, sizeof(char) * len_amino_seq))
	CHECK_CUDA(cudaMalloc((void**)&d_pop, sizeof(char) * numBlocks * len_sol * 2))
	CHECK_CUDA(cudaMalloc((void**)&d_objval, sizeof(float) * numBlocks * OBJECTIVE_NUM * 2))
	CHECK_CUDA(cudaMalloc((void**)&d_objidx, sizeof(char) * numBlocks * OBJECTIVE_NUM * 2 * 2))
	CHECK_CUDA(cudaMalloc((void**)&d_lrcsval, sizeof(int) * numBlocks * 3 * 2))
	CHECK_CUDA(cudaMalloc((void**)&d_sorted_array, sizeof(int) * numBlocks * 2))
	CHECK_CUDA(cudaMalloc((void**)&d_F_set, sizeof(int) * numBlocks * 2 * numBlocks * 2))
	CHECK_CUDA(cudaMalloc((void**)&d_Sp_set, sizeof(int) * numBlocks * 2 * numBlocks * 2))
	CHECK_CUDA(cudaMalloc((void**)&d_Q_set, sizeof(int) * numBlocks * 2))
	CHECK_CUDA(cudaMalloc((void**)&d_np, sizeof(int) * numBlocks * 2))



	/* memory copy host to device */
	CHECK_CUDA(cudaMemcpy(d_amino_seq_idx, h_amino_seq_idx, sizeof(char) * len_amino_seq, cudaMemcpyHostToDevice))
	CHECK_CUDA(cudaMemcpyToSymbol(c_codons_weight, Codons_weight, sizeof(Codons_weight)))
	CHECK_CUDA(cudaMemcpyToSymbol(c_amino_startpos, h_amino_startpos, sizeof(char) * 20))
	CHECK_CUDA(cudaMemcpyToSymbol(c_codons, Codons, sizeof(Codons)))
	CHECK_CUDA(cudaMemcpyToSymbol(c_codons_num, Codons_num, sizeof(Codons_num)))
	CHECK_CUDA(cudaMemcpyToSymbol(c_len_amino_seq, &len_amino_seq, sizeof(int)))
	CHECK_CUDA(cudaMemcpyToSymbol(c_cds_num, &cds_num, sizeof(int)))
	CHECK_CUDA(cudaMemcpyToSymbol(c_mprob, &mprob, sizeof(float)))


	/* ------------------------------------------------ kerenl call ----------------------------------------------- */
	setup_kernel << <numBlocks, threadsPerBlock >> > (genState, rand());

	CHECK_CUDA(cudaEventRecord(d_start))
	GenSolution << <numBlocks, threadsPerBlock, sizeof(int)* (threadsPerBlock + 3) + sizeof(float) * (threadsPerBlock + OBJECTIVE_NUM) + sizeof(char) * (len_sol + len_amino_seq + 2 * OBJECTIVE_NUM) >> >
		(genState, d_amino_seq_idx, d_pop, d_objval, d_objidx, d_lrcsval, d_sorted_array);
	CHECK_CUDA(cudaEventRecord(d_end))
	CHECK_CUDA(cudaEventSynchronize(d_end))
	CHECK_CUDA(cudaEventElapsedTime(&kernel_time, d_start, d_end))
	printf("\nGenerating solution kerenl time : %f seconds\n", kernel_time / 1000.f);



	/* For check sorting */
	int numBlocksPerSm = 0;
	CHECK_CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, FastSortSolution, threadsPerBlock, 0))
	printf("\nFor sorting numBlockPerSm : %d\n", numBlocksPerSm);
	void* args[] = {&d_sorted_array, &d_objval, &d_F_set, &d_Sp_set, &d_Q_set, &d_np};
	k = (total_cycle % sorting_cycle == 0) ? total_cycle / sorting_cycle : total_cycle / sorting_cycle + 1;
	CHECK_CUDA(cudaEventRecord(d_start))
	for (i = 0; i < k; i++)
	{
		if (i == k - 1 && (total_cycle % sorting_cycle != 0)) {
			mainKernel << <numBlocks, threadsPerBlock, sizeof(int)* (threadsPerBlock + 3 * 2) + sizeof(float) * (threadsPerBlock + OBJECTIVE_NUM * 2) + sizeof(char) * (len_sol * 2 + len_amino_seq + OBJECTIVE_NUM * 2 * 2 + 1) >> >
				(genState, d_amino_seq_idx, d_pop, d_objval, d_objidx, d_lrcsval, total_cycle % sorting_cycle, d_sorted_array);
		}
		else {
			mainKernel << <numBlocks, threadsPerBlock, sizeof(int)* (threadsPerBlock + 3 * 2) + sizeof(float) * (threadsPerBlock + OBJECTIVE_NUM * 2) + sizeof(char) * (len_sol * 2 + len_amino_seq + OBJECTIVE_NUM * 2 * 2 + 1) >> >
				(genState, d_amino_seq_idx, d_pop, d_objval, d_objidx, d_lrcsval, sorting_cycle, d_sorted_array);
		}
		
		//	sorting
		//CHECK_CUDA(cudaLaunchCooperativeKernel((void*)FastSortSolution, numBlocksPerSm, threadsPerBlock, args, 0))
	}
	CHECK_CUDA(cudaEventRecord(d_end))
	CHECK_CUDA(cudaEventSynchronize(d_end))
	CHECK_CUDA(cudaEventElapsedTime(&kernel_time, d_start, d_end))
	printf("Mutation kernel + Sort kernel time : %f seconds\n", kernel_time / 1000.f);
	//printf("using shared memory size : %lu\n", sizeof(int) * (threadsPerBlock + 3 * 2) + sizeof(float) * (threadsPerBlock + OBJECTIVE_NUM * 2) + sizeof(char) * (len_sol * 2 + len_amino_seq + OBJECTIVE_NUM * 2 * 2 + 1));
	printf("using contant memory size : %lu\n", sizeof(Codons_weight) + sizeof(char) * 20 + sizeof(Codons) + sizeof(Codons_num) + sizeof(int) * 2 + sizeof(float));




	/* memory copy device to host */
	CHECK_CUDA(cudaMemcpy(h_pop, d_pop, sizeof(char) * numBlocks * len_sol * 2, cudaMemcpyDeviceToHost))
	CHECK_CUDA(cudaMemcpy(h_objval, d_objval, sizeof(float) * numBlocks * OBJECTIVE_NUM * 2, cudaMemcpyDeviceToHost))
	CHECK_CUDA(cudaMemcpy(h_objidx, d_objidx, sizeof(char) * numBlocks * OBJECTIVE_NUM * 2 * 2, cudaMemcpyDeviceToHost))
	CHECK_CUDA(cudaMemcpy(h_lrcsval, d_lrcsval, sizeof(int) * numBlocks * 3 * 2, cudaMemcpyDeviceToHost))
	CHECK_CUDA(cudaMemcpy(h_sorted_array, d_sorted_array, sizeof(int) * numBlocks * 2, cudaMemcpyDeviceToHost))


	// print minimum distance to ideal point
	min_dist = MinEuclid(h_objval, pop_size * 2);
	printf("minimum distance to the ideal point : %f\n", min_dist);
	//printf("lowest mcai value : %f\n", lowest_mcai);

	printf("\nSorted array index : \n");
	for (i = 0; i < pop_size; i++) {
		printf("%d ", h_sorted_array[i]);
	}

	/* print solution */
	//for (i = 0; i < pop_size * 2; i++)
	//{
	//	printf("%d solution\n", i + 1);
	//	for (j = 0; j < cds_num; j++) {
	//		printf("%d cds : ", j + 1);
	//		for (k = 0; k < len_cds; k++) {
	//			printf("%c", h_pop[len_sol * i + len_cds * j + k]);
	//		}
	//		printf("\n");
	//	}
	//	printf("\n");
	//}

	/* print objective value */
	printf("\n\n");
	for (i = 0; i < pop_size * 2; i++)
	{
		printf("%d solution\n", i + 1);
		printf("mCAI : %f mHD : %f MLRCS : %f\n", h_objval[i * OBJECTIVE_NUM + _mCAI], h_objval[i * OBJECTIVE_NUM + _mHD], h_objval[i * OBJECTIVE_NUM + _MLRCS]);
		printf("mCAI idx : %d mHD idx : %d %d MLRCS idx : %d %d\n", h_objidx[i * OBJECTIVE_NUM * 2 + _mCAI * 2],
			h_objidx[i * OBJECTIVE_NUM * 2 + _mHD * 2], h_objidx[i * OBJECTIVE_NUM * 2 + _mHD * 2 + 1],
			h_objidx[i * OBJECTIVE_NUM * 2 + _MLRCS * 2], h_objidx[i * OBJECTIVE_NUM * 2 + _MLRCS * 2 + 1]);
		printf("P : %d Q : %d L : %d\n", h_lrcsval[i * 3 + P], h_lrcsval[i * 3 + Q], h_lrcsval[i * 3 + L]);
	}


	fp = fopen("test.txt", "w");
	/* for computing hypervolume write file */
	for (i = 0; i < pop_size * 2; i++)
	{
		fprintf(fp, "%f %f %f\n", -h_objval[i * OBJECTIVE_NUM + _mCAI], -h_objval[i * OBJECTIVE_NUM + _mHD] / 0.4, h_objval[i * OBJECTIVE_NUM + _MLRCS]);
	}
	fclose(fp);




	/* free deivce memory */
	CHECK_CUDA(cudaFree(genState))
	CHECK_CUDA(cudaFree(d_amino_seq_idx))
	CHECK_CUDA(cudaFree(d_pop))
	CHECK_CUDA(cudaFree(d_objval))
	CHECK_CUDA(cudaFree(d_objidx))
	CHECK_CUDA(cudaFree(d_lrcsval))
	CHECK_CUDA(cudaFree(d_sorted_array))
	CHECK_CUDA(cudaFree(d_F_set))
	CHECK_CUDA(cudaFree(d_Sp_set))
	CHECK_CUDA(cudaFree(d_Q_set))
	CHECK_CUDA(cudaFree(d_np))
	CHECK_CUDA(cudaEventDestroy(d_start))
	CHECK_CUDA(cudaEventDestroy(d_end))

	/* free host memory */
	free(amino_seq);
	free(h_amino_seq_idx);
	free(h_amino_startpos);
	free(h_pop);
	free(h_objval);
	free(h_objidx);
	free(h_lrcsval);
	free(h_sorted_array);


	return EXIT_SUCCESS;
}
