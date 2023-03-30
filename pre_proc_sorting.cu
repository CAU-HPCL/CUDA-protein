/* include C/C++ header */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <chrono>

/* include CUDA header */
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <curand_kernel.h>
#include <cuda.h>
//#include <cooperative_groups.h>

#define _CRT_SECURE_NO_WARINGS

#define CHECK_CUDA(func)                                               \
	{                                                                  \
		cudaError_t status = (func);                                   \
		if (status != cudaSuccess)                                     \
		{                                                              \
			printf("CUDA API failed at line %d with error: %s (%d)\n", \
				   __LINE__, cudaGetErrorString(status), status);      \
			return EXIT_FAILURE;                                       \
		}                                                              \
	}

#define WARP_SIZE 32

#define NOT_FOUND -1

#define RANDOM 0
#define UPPER 1
#define LOWER 2

#define CODON_SIZE 3
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
#define IDEAL_MHD 1
#define IDEAL_MLRCS 0
#define EUCLID(val1, val2, val3) (float)sqrt(pow(IDEAL_MCAI - val1, 2) + pow(IDEAL_MHD - val2, 2) + pow(val3, 2))

/* -------------------- 20 kinds of amino acids & weights are sorted ascending order -------------------- */
char Amino_abbreviation[20] = {'A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y'};
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
char Codons_num[20] = {4, 2, 2, 2, 2, 4, 2, 3, 2, 6, 1, 2, 4, 2, 6, 6, 4, 4, 1, 2};
float Codons_weight[61] = {1854 / 13563.0f, 5296 / 13563.0f, 7223 / 135063.0f, 1.0f,
						   1234 / 3052.0f, 1.0f,
						   8960 / 12731.0f, 1.0f,
						   6172 / 19532.0f, 1.0f,
						   7773 / 8251.0f, 1.0f,
						   1852 / 15694.0f, 2781 / 15694.0f, 3600 / 15694.0f, 1.0f,
						   3288 / 4320.0f, 1.0f,
						   3172 / 12071.0f, 8251 / 12071.0f, 1.0f,
						   12845 / 15169.0f, 1.0f,
						   1242 / 13329.0f, 2852 / 13329.0f, 3207 / 13329.0f, 4134 / 13329.0f, 8549 / 13329.0f, 1.0f,
						   1.0f,
						   8613 / 9875.0f, 1.0f,
						   1064 / 8965.0f, 1656 / 8965.0f, 4575 / 8965.0f, 1.0f,
						   3312 / 10987.0f, 1.0f,
						   342 / 9784.0f, 489 / 9784.0f, 658 / 9784.0f, 2175 / 9784.0f, 3307 / 9784.0f, 1.0f,
						   2112 / 10025.0f, 2623 / 10025.0f, 3873 / 10025.0f, 4583 / 10025.0f, 6403 / 10025.0f, 1.0f,
						   1938 / 9812.0f, 5037 / 9812.0f, 6660 / 9812.0f, 1.0f,
						   3249 / 11442.0f, 3700 / 11442.0f, 6911 / 11442.0f, 1.0f,
						   1.0f,
						   5768 / 7114.0f, 1.0f};
/* ------------------------------ end of definition ------------------------------ */

/* find index of Amino_abbreviation array matching with input amino abbreviation using binary search */
__host__ int FindAminoIndex(char amino_abbreviation)
{
	int low = 0;
	int high = 20 - 1;
	int mid;

	while (low <= high)
	{
		mid = (low + high) / 2;

		if (Amino_abbreviation[mid] == amino_abbreviation)
			return mid;
		else if (Amino_abbreviation[mid] > amino_abbreviation)
			high = mid - 1;
		else
			low = mid + 1;
	}

	printf("FindAminoIndex function failure\n");
	exit(EXIT_FAILURE);
}

/* Minimum distance to optimal objective value(point) */
__host__ float MinEuclid(const float *objval, int pop_size)
{
	float res;
	float tmp;

	res = EUCLID(objval[_mCAI], objval[_mHD], objval[_MLRCS]);
	for (int i = 0; i < pop_size; i++)
	{
		tmp = EUCLID(objval[i * OBJECTIVE_NUM + _mCAI], objval[i * OBJECTIVE_NUM + _mHD], objval[i * OBJECTIVE_NUM + _MLRCS]);
		if (tmp < res)
			res = tmp;
	}

	return res;
}

__constant__ char c_amino_startpos[20];
__constant__ char c_codons[61 * CODON_SIZE + 1];
__constant__ char c_codons_num[20];
__constant__ int c_len_amino_seq;
__constant__ int c_cds_num;
__constant__ int c_sort_popsize;
__constant__ float c_codons_weight[61];
__constant__ float c_mprob;

__device__ int lock = 0; // for atomic operation
__device__ int front = 0;
__device__ int count = 0;
__device__ int sorting_idx = 0;

__device__ char FindNum_C(const char *origin, const char *target, char num_codons)
{
	for (char i = 0; i < num_codons; i++)
	{
		if (target[0] == origin[i * CODON_SIZE] && target[1] == origin[i * CODON_SIZE + 1] && target[2] == origin[i * CODON_SIZE + 2])
		{
			return i;
		}
	}

	printf("FindNum_C function failure\n");
}

/* mutate codon upper adaptation or randmom adaptation */
__device__ void mutation(curandStateXORWOW *state, const char *codon_info, char *target, char total_num, char origin_pos, const float mprob, const int type)
{
	float cd_prob;
	char new_idx;

	/* 1.0 is included and 0.0 is excluded */
	cd_prob = curand_uniform(state);

	switch (type)
	{
	case RANDOM:
		new_idx = (char)(curand_uniform(state) * total_num);
		if (cd_prob <= mprob && total_num > 1)
		{
			while (origin_pos == new_idx || new_idx == total_num)
			{
				new_idx = (char)(curand_uniform(state) * total_num);
			}
			target[0] = codon_info[new_idx * CODON_SIZE];
			target[1] = codon_info[new_idx * CODON_SIZE + 1];
			target[2] = codon_info[new_idx * CODON_SIZE + 2];
		}
		break;

	case UPPER:
		new_idx = (char)(curand_uniform(state) * (total_num - 1 - origin_pos));
		if (cd_prob <= mprob && (origin_pos != (total_num - 1)))
		{
			while (new_idx == (total_num - 1 - origin_pos))
			{
				new_idx = (char)(curand_uniform(state) * (total_num - 1 - origin_pos));
			}
			target[0] = codon_info[(origin_pos + 1 + new_idx) * CODON_SIZE];
			target[1] = codon_info[(origin_pos + 1 + new_idx) * CODON_SIZE + 1];
			target[2] = codon_info[(origin_pos + 1 + new_idx) * CODON_SIZE + 2];
		}
		break;

		case LOWER:
			new_idx = (char)(curand_uniform(state) * origin_pos);
			if (cd_prob <= mprob && origin_pos != 0) {
				while (new_idx == origin_pos) {
					new_idx = (char)(curand_uniform(state) * origin_pos);
				}
				target[0] = codon_info[new_idx * CODON_SIZE];
				target[1] = codon_info[new_idx * CODON_SIZE + 1];
				target[2] = codon_info[new_idx * CODON_SIZE + 2];
			}
			break;
	}

	return;
}

__device__ bool ParetoComparison(const float *new_objval, const float *old_objval)
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
__global__ void setup_kernel(curandStateXORWOW *state, unsigned long long seed)
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
__global__ void GenSolution(curandStateXORWOW* state, const char* d_amino_seq_idx, char* d_pop, float* d_objval, char* d_objidx, int* d_lrcsval, int* d_sorted_array, const int limit, const float lowest_mcai)
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


	float section_low, section_high, adjust_prob;
	char direct;
	int cnt;


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


	/* -------------------- initialize solution -------------------- */

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
		while (true) {
			if (threadIdx.x < j) {
				s_obj_compute[threadIdx.x] *= s_obj_compute[threadIdx.x + j];
			}
			__syncthreads();

			if (j == 1)
				break;

			if ((j % 2 == 1) && (threadIdx.x == 0))
			{
				s_obj_compute[0] *= s_obj_compute[j - 1];
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



	/* ------------------------------ intentional mutation initail solution for adjusting mCAI ------------------------------ */
	section_low = lowest_mcai + (1 - lowest_mcai) / gridDim.x * (blockIdx.x % gridDim.x);
	section_high = lowest_mcai + (1 - lowest_mcai) / gridDim.x * (blockIdx.x % gridDim.x + 1);

	/* muatate */
	adjust_prob = 1.f;
	cnt = 0;
	// mutate direction
	if (s_sol_objval[_mCAI] < section_low)
		direct = UPPER;
	else if (s_sol_objval[_mCAI] > section_high)
		direct = LOWER;
	if (blockIdx.x != gridDim.x - 1) {
		while (cnt < limit && (s_sol_objval[_mCAI] < section_low || s_sol_objval[_mCAI] > section_high))
		{
			if (s_sol_objval[_mCAI] < section_low) {
				if (direct != UPPER) {
					direct = UPPER;
					adjust_prob /= 2;
				}
				num_partition = ((c_len_amino_seq * c_cds_num) % blockDim.x == 0) ? (c_len_amino_seq * c_cds_num) / blockDim.x : (c_len_amino_seq * c_cds_num) / blockDim.x + 1;
				for (i = 0; i < num_partition; i++) {
					idx = blockDim.x * i + threadIdx.x;
					if (idx < c_len_amino_seq * c_cds_num) {
						seq_idx = idx % c_len_amino_seq;

						pos = FindNum_C(&c_codons[c_amino_startpos[s_amino_seq_idx[seq_idx]] * CODON_SIZE], &s_sol[idx * CODON_SIZE],
							c_codons_num[s_amino_seq_idx[seq_idx]]);
						mutation(&localState, &c_codons[c_amino_startpos[s_amino_seq_idx[seq_idx]] * CODON_SIZE], &s_sol[idx * CODON_SIZE],
							c_codons_num[s_amino_seq_idx[seq_idx]], pos, adjust_prob, UPPER);
					}
				}
			}
			else {
				if (direct != LOWER) {
					direct = LOWER;
					adjust_prob /= 2;
				}
				num_partition = ((c_len_amino_seq * c_cds_num) % blockDim.x == 0) ? (c_len_amino_seq * c_cds_num) / blockDim.x : (c_len_amino_seq * c_cds_num) / blockDim.x + 1;
				for (i = 0; i < num_partition; i++) {
					idx = blockDim.x * i + threadIdx.x;
					if (idx < c_len_amino_seq * c_cds_num) {
						seq_idx = idx % c_len_amino_seq;

						pos = FindNum_C(&c_codons[c_amino_startpos[s_amino_seq_idx[seq_idx]] * CODON_SIZE], &s_sol[idx * CODON_SIZE],
							c_codons_num[s_amino_seq_idx[seq_idx]]);
						mutation(&localState, &c_codons[c_amino_startpos[s_amino_seq_idx[seq_idx]] * CODON_SIZE], &s_sol[idx * CODON_SIZE],
							c_codons_num[s_amino_seq_idx[seq_idx]], pos, adjust_prob, LOWER);
					}
				}
			}

			/* calculate mCAI value */
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
				while (true) {
					if (threadIdx.x < j) {
						s_obj_compute[threadIdx.x] *= s_obj_compute[threadIdx.x + j];
					}
					__syncthreads();

					if (j == 1)
						break;

					if ((j % 2 == 1) && (threadIdx.x == 0))
					{
						s_obj_compute[0] *= s_obj_compute[j - 1];
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
			cnt++;
		}
	}
	/* ------------------------------ end of intational muation ------------------------------ */



	/* calculate mHD */
	num_partition = (len_cds % blockDim.x == 0) ? (len_cds / blockDim.x) : (len_cds / blockDim.x) + 1;
	for (i = 0; i < c_cds_num - 1; i++) {
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
			while (true) {
				if (threadIdx.x < k) {
					s_obj_compute[threadIdx.x] += s_obj_compute[threadIdx.x + k];
				}
				__syncthreads();

				if (k == 1)
					break;

				if ((k % 2 == 1) && (threadIdx.x == 0))
				{
					s_obj_compute[0] += s_obj_compute[k - 1];
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
	while (true)
	{
		if (threadIdx.x < j && (s_obj_compute[threadIdx.x + j] > s_obj_compute[threadIdx.x]))
		{
			s_obj_compute[threadIdx.x] = s_obj_compute[threadIdx.x + j];
			s_lrcs_tid[threadIdx.x] = s_lrcs_tid[threadIdx.x + j];
		}
		__syncthreads();

		if (j == 1)
			break;

		if ((j % 2 == 1) && (threadIdx.x == 0))
		{
			if (s_obj_compute[j - 1] > s_obj_compute[0]) {
				s_obj_compute[0] = s_obj_compute[j - 1];
				s_lrcs_tid[0] = s_lrcs_tid[j - 1];
			}
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
	/* -------------------- end of initialize -------------------- */


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

typedef struct
{
	int sol_idx;
	float corwding_dist;
	float obj_val[OBJECTIVE_NUM];
} Sol;

__device__ void Sol_assign(Sol *s1, Sol *s2)
{
	int i;

	s1->corwding_dist = s2->corwding_dist;
	s1->sol_idx = s2->sol_idx;
	for (i = 0; i < OBJECTIVE_NUM; i++)
	{
		s1->obj_val[i] = s2->obj_val[i];
	}

	return;
}

__device__ void CompUp(Sol *s1, Sol *s2, int idx)
{
	Sol tmp;

	if (s1->obj_val[idx] > s2->obj_val[idx])
	{
		Sol_assign(&tmp, s1);
		Sol_assign(s1, s2);
		Sol_assign(s2, &tmp);
	}
	return;
}

__device__ void CompDownCrowd(Sol *s1, Sol *s2)
{
	Sol tmp;

	if (s1->corwding_dist < s2->corwding_dist)
	{
		Sol_assign(&tmp, s1);
		Sol_assign(s1, s2);
		Sol_assign(s2, &tmp);
	}

	return;
}

/*
Based on sorting methods on NSGA2 paper
If we input number of threads and sorting function into cudaOccupancyMaxActiveBlocksPerMultiprocessor(), we get number of blocks available to use
*/
__global__ void FastSortSolution(int *d_sorted_array, bool *F_set, bool *Sp_set, char *d_pop, float *d_objval, char *d_objidx, int *d_lrcsval)
{
	int i, j;
	int sol_idx;
	int sec1, sec2;

	extern __shared__ int smem[];
	__shared__ Sol *s_sol_struct;
	__shared__ int *s_rank_count;
	__shared__ int *s_np;
	__shared__ float *s_objval;

	s_sol_struct = (Sol *)smem;
	s_rank_count = (int *)&s_sol_struct[blockDim.x];
	s_np = (int *)&s_rank_count[blockDim.x];
	s_objval = (float *)&s_np[blockDim.x];

	s_rank_count[threadIdx.x] = 0;
	s_np[threadIdx.x] = 0;

	/* copy objective value from global memory to shared memory */
	s_objval[threadIdx.x * OBJECTIVE_NUM + _mCAI] = d_objval[threadIdx.x * OBJECTIVE_NUM + _mCAI];
	s_objval[threadIdx.x * OBJECTIVE_NUM + _mHD] = d_objval[threadIdx.x * OBJECTIVE_NUM + _mHD];
	s_objval[threadIdx.x * OBJECTIVE_NUM + _MLRCS] = d_objval[threadIdx.x * OBJECTIVE_NUM + _MLRCS];

	if (threadIdx.x == 0)
	{
		front = 0;
		count = 0;
		sorting_idx = 0;
	}
	__syncthreads();

	/* -------------------- 1st front setting -------------------- */
	for (i = 0; i < c_sort_popsize; i++)
	{
		if (threadIdx.x != i)
		{
			if (ParetoComparison(&s_objval[threadIdx.x * OBJECTIVE_NUM], &s_objval[i * OBJECTIVE_NUM]))
				Sp_set[threadIdx.x * c_sort_popsize + i] = true;
			else if (ParetoComparison(&s_objval[i * OBJECTIVE_NUM], &s_objval[threadIdx.x * OBJECTIVE_NUM]))
				s_np[threadIdx.x] += 1;
		}
	}
	if (s_np[threadIdx.x] == 0)
	{
		F_set[threadIdx.x] = true;
		while (atomicCAS(&lock, 0, 1) != 0)
			;
		d_sorted_array[count] = threadIdx.x;
		count += 1;
		s_rank_count[front] += 1;
		atomicExch(&lock, 0);
	}
	__syncthreads();


	sol_idx = 0;
	// crowding distance sort
	if (count > (c_sort_popsize / 2))
	{
		// write solution to shared memory Sol
		if (F_set[threadIdx.x])
		{
			while (atomicCAS(&lock, 0, 1) != 0)
				;
			sol_idx = sorting_idx++;
			atomicExch(&lock, 0);
			s_sol_struct[sol_idx].sol_idx = threadIdx.x;
			s_sol_struct[sol_idx].corwding_dist = 0;
			s_sol_struct[sol_idx].obj_val[_mCAI] = d_objval[threadIdx.x * OBJECTIVE_NUM + _mCAI];
			s_sol_struct[sol_idx].obj_val[_mHD] = d_objval[threadIdx.x * OBJECTIVE_NUM + _mHD] / IDEAL_MHD;
			s_sol_struct[sol_idx].obj_val[_MLRCS] = d_objval[threadIdx.x * OBJECTIVE_NUM + _MLRCS];
		}
		__syncthreads();

		for (i = 0; i < OBJECTIVE_NUM; i++)
		{
			// sorting objective function ascending order
			sec1 = 1;
			while (sec1 < s_rank_count[front])
			{
				if ((threadIdx.x % (sec1 * 2) < sec1) && ((sec1 * 2 * (threadIdx.x / (sec1 * 2) + 1) - threadIdx.x % (sec1 * 2) - 1) < s_rank_count[front]))
					CompUp(&s_sol_struct[threadIdx.x], &s_sol_struct[sec1 * 2 * (threadIdx.x / (sec1 * 2) + 1) - (threadIdx.x % (sec1 * 2)) - 1], i);

				sec2 = sec1 / 2;
				__syncthreads();

				while (sec2 != 0)
				{
					if ((threadIdx.x % (sec2 * 2) < sec2) && (threadIdx.x + sec2 < s_rank_count[front]))
						CompUp(&s_sol_struct[threadIdx.x], &s_sol_struct[threadIdx.x + sec2], i);
					sec2 /= 2;
					__syncthreads();
				}

				sec1 *= 2;
			}
			__syncthreads();

			if (threadIdx.x < s_rank_count[front])
			{
				if (threadIdx.x == 0)
					s_sol_struct[threadIdx.x].corwding_dist = 10000.f;
				else if (threadIdx.x == s_rank_count[front] - 1)
					s_sol_struct[threadIdx.x].corwding_dist = 10000.f;
				else
					s_sol_struct[threadIdx.x].corwding_dist += s_sol_struct[threadIdx.x + 1].obj_val[i] - s_sol_struct[threadIdx.x - 1].obj_val[i];
			}
			__syncthreads();
		}

		// sort crowding distance descending order
		sec1 = 1;
		while (sec1 < s_rank_count[front])
		{
			if ((threadIdx.x % (sec1 * 2)) < sec1 && ((sec1 * 2 * (threadIdx.x / (sec1 * 2) + 1) - threadIdx.x % (sec1 * 2) - 1) < s_rank_count[front]))
				CompDownCrowd(&s_sol_struct[threadIdx.x], &s_sol_struct[sec1 * 2 * (threadIdx.x / (sec1 * 2) + 1) - (threadIdx.x % (sec1 * 2)) - 1]);

			sec2 = sec1 / 2;
			__syncthreads();

			while (sec2 != 0)
			{
				if ((threadIdx.x % (sec2 * 2) < sec2) && (threadIdx.x + sec2 < s_rank_count[front]))
					CompDownCrowd(&s_sol_struct[threadIdx.x], &s_sol_struct[threadIdx.x + sec2]);
				sec2 /= 2;
				__syncthreads();
			}

			sec1 *= 2;
		}
		__syncthreads();

		if (threadIdx.x < s_rank_count[front])
		{
			d_sorted_array[count - s_rank_count[front] + threadIdx.x] = s_sol_struct[threadIdx.x].sol_idx;
		}

		return;
	}


	/* -------------------- non dominated sort  -------------------- */
	if (threadIdx.x == 0)
		front += 1;
	__syncthreads();
	for (i = 0; i < c_sort_popsize - 1; i++)
	{
		for (j = 0; j < c_sort_popsize; j++)
		{
			if (F_set[(front - 1) * c_sort_popsize + j] && Sp_set[j * c_sort_popsize + threadIdx.x])
			{
				s_np[threadIdx.x] -= 1;
				if (s_np[threadIdx.x] == 0)
				{
					F_set[front * c_sort_popsize + threadIdx.x] = true;
					while (atomicCAS(&lock, 0, 1) != 0)
						;
					d_sorted_array[count] = threadIdx.x;
					count += 1;
					s_rank_count[front] += 1; // rank_count 는 체크를 위한 부분으로 나중에 빠지는 것 가능!
					atomicExch(&lock, 0);
				}
			}
		}
		__syncthreads();

		if (count > (c_sort_popsize / 2))
			break;

		if (threadIdx.x == 0)
			front += 1;
		__syncthreads();
	}

	sol_idx = 0;
	// write solution to shared memory Sol
	if (F_set[front * c_sort_popsize + threadIdx.x])
	{
		while (atomicCAS(&lock, 0, 1) != 0)
			;
		sol_idx = sorting_idx++;
		atomicExch(&lock, 0);
		s_sol_struct[sol_idx].sol_idx = threadIdx.x;
		s_sol_struct[sol_idx].corwding_dist = 0;
		s_sol_struct[sol_idx].obj_val[_mCAI] = d_objval[threadIdx.x * OBJECTIVE_NUM + _mCAI];
		s_sol_struct[sol_idx].obj_val[_mHD] = d_objval[threadIdx.x * OBJECTIVE_NUM + _mHD] / IDEAL_MHD;
		s_sol_struct[sol_idx].obj_val[_MLRCS] = d_objval[threadIdx.x * OBJECTIVE_NUM + _MLRCS];
	}
	__syncthreads();

	for (i = 0; i < OBJECTIVE_NUM; i++)
	{
		// sorting objective function ascending order
		sec1 = 1;
		while (sec1 < s_rank_count[front])
		{
			if ((threadIdx.x % (sec1 * 2) < sec1) && ((sec1 * 2 * (threadIdx.x / (sec1 * 2) + 1) - threadIdx.x % (sec1 * 2) - 1) < s_rank_count[front]))
				CompUp(&s_sol_struct[threadIdx.x], &s_sol_struct[sec1 * 2 * (threadIdx.x / (sec1 * 2) + 1) - (threadIdx.x % (sec1 * 2)) - 1], i);

			sec2 = sec1 / 2;
			__syncthreads();
			while (sec2 != 0)
			{
				if ((threadIdx.x % (sec2 * 2) < sec2) && (threadIdx.x + sec2 < s_rank_count[front]))
					CompUp(&s_sol_struct[threadIdx.x], &s_sol_struct[threadIdx.x + sec2], i);
				sec2 /= 2;
				__syncthreads();
			}

			sec1 *= 2;
		}
		__syncthreads();

		if (threadIdx.x < s_rank_count[front])
		{
			if (threadIdx.x == 0)
				s_sol_struct[threadIdx.x].corwding_dist = 10000.f;
			else if (threadIdx.x == s_rank_count[front] - 1)
				s_sol_struct[threadIdx.x].corwding_dist = 10000.f;
			else
				s_sol_struct[threadIdx.x].corwding_dist += s_sol_struct[threadIdx.x + 1].obj_val[i] - s_sol_struct[threadIdx.x - 1].obj_val[i];
		}
		__syncthreads();
	}

	// sort crowding distance descending order
	sec1 = 1;
	while (sec1 < s_rank_count[front])
	{
		if ((threadIdx.x % (sec1 * 2)) < sec1 && ((sec1 * 2 * (threadIdx.x / (sec1 * 2) + 1) - threadIdx.x % (sec1 * 2) - 1) < s_rank_count[front]))
			CompDownCrowd(&s_sol_struct[threadIdx.x], &s_sol_struct[sec1 * 2 * (threadIdx.x / (sec1 * 2) + 1) - (threadIdx.x % (sec1 * 2)) - 1]);
		__syncthreads();

		sec2 = sec1 / 2;
		while (sec2 != 0)
		{
			if ((threadIdx.x % (sec2 * 2) < sec2) && (threadIdx.x + sec2 < s_rank_count[front]))
				CompDownCrowd(&s_sol_struct[threadIdx.x], &s_sol_struct[threadIdx.x + sec2]);
			sec2 /= 2;
			__syncthreads();
		}
		sec1 *= 2;
	}
	__syncthreads();

	if (threadIdx.x < s_rank_count[front])
	{
		d_sorted_array[count - s_rank_count[front] + threadIdx.x] = s_sol_struct[threadIdx.x].sol_idx;
	}

	return;
}

/*
copy solution from global memory based on sorted array
Not update solution to global memory state of sorted Just write solution to global memory
This means In global memory solution is not sorted after this function
If you want soltion is sorted sorted function call and we get sorted array and update solution based on sorted array
*/
__global__ void mainKernel(curandStateXORWOW *state, const char *d_amino_seq_idx, char *d_pop, float *d_objval, char *d_objidx, int *d_lrcsval, const int cycle, char *tmp_pop, float *tmp_objval, char *tmp_objidx, int *tmp_lrcsval, int *d_sorted_array)
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

	char *ptr_origin_sol, *ptr_target_sol;
	float *ptr_origin_objval, *ptr_target_objval;
	char *ptr_origin_objidx, *ptr_target_objidx;
	int *ptr_origin_lrcsval, *ptr_target_lrcsval; // P, Q, L

	id = blockDim.x * blockIdx.x + threadIdx.x;
	localState = state[id];
	len_cds = c_len_amino_seq * CODON_SIZE;
	len_sol = len_cds * c_cds_num;

	/* -------------------- shared memory allocation -------------------- */
	extern __shared__ int smem[];
	__shared__ char *s_amino_seq_idx;
	__shared__ char *s_sol1;
	__shared__ char *s_sol2;
	__shared__ char *s_sol1_objidx;
	__shared__ char *s_sol2_objidx;
	__shared__ char *mutation_type;
	__shared__ float *s_obj_compute; // for computing mCAI & mHD value
	__shared__ float *s_sol1_objval;
	__shared__ float *s_sol2_objval;
	__shared__ int *s_sol1_lrcsval;
	__shared__ int *s_sol2_lrcsval;
	__shared__ int *s_lrcs_tid;

	s_lrcs_tid = smem;
	s_sol1_lrcsval = (int *)&s_lrcs_tid[blockDim.x]; // for finding which thread have LRCS
	s_sol2_lrcsval = (int *)&s_sol1_lrcsval[3];
	s_obj_compute = (float *)&s_sol2_lrcsval[3];
	s_sol1_objval = (float *)&s_obj_compute[blockDim.x];
	s_sol2_objval = (float *)&s_sol1_objval[OBJECTIVE_NUM];
	s_amino_seq_idx = (char *)&s_sol2_objval[OBJECTIVE_NUM];
	s_sol1 = (char *)&s_amino_seq_idx[c_len_amino_seq];
	s_sol2 = (char *)&s_sol1[len_sol];
	s_sol1_objidx = (char *)&s_sol2[len_sol];
	s_sol2_objidx = (char *)&s_sol1_objidx[OBJECTIVE_NUM * 2];
	mutation_type = (char *)&s_sol2_objidx[OBJECTIVE_NUM * 2];
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

	num_partition = (c_len_amino_seq % blockDim.x == 0) ? (c_len_amino_seq / blockDim.x) : (c_len_amino_seq / blockDim.x) + 1;
	for (i = 0; i < num_partition; i++)
	{
		idx = blockDim.x * i + threadIdx.x;
		if (idx < c_len_amino_seq)
		{
			s_amino_seq_idx[idx] = d_amino_seq_idx[idx];
		}
	}
	// -----------------------------------------------------------------------------------------

	/* copy solution from global memory to shared memory based on sorted array index */
	num_partition = (len_sol % blockDim.x == 0) ? (len_sol / blockDim.x) : (len_sol / blockDim.x) + 1;
	for (i = 0; i < num_partition; i++)
	{
		idx = blockDim.x * i + threadIdx.x;
		if (idx < len_sol)
			ptr_origin_sol[idx] = tmp_pop[d_sorted_array[blockIdx.x] * len_sol + idx];
	}

	if (threadIdx.x == 0)
	{
		ptr_origin_objval[_mCAI] = tmp_objval[d_sorted_array[blockIdx.x] * OBJECTIVE_NUM + _mCAI];
		ptr_origin_objval[_mHD] = tmp_objval[d_sorted_array[blockIdx.x] * OBJECTIVE_NUM + _mHD];
		ptr_origin_objval[_MLRCS] = tmp_objval[d_sorted_array[blockIdx.x] * OBJECTIVE_NUM + _MLRCS];

		ptr_origin_objidx[_mCAI * 2] = tmp_objidx[d_sorted_array[blockIdx.x] * OBJECTIVE_NUM * 2 + _mCAI * 2];
		ptr_origin_objidx[_mHD * 2] = tmp_objidx[d_sorted_array[blockIdx.x] * OBJECTIVE_NUM * 2 + _mHD * 2];
		ptr_origin_objidx[_mHD * 2 + 1] = tmp_objidx[d_sorted_array[blockIdx.x] * OBJECTIVE_NUM * 2 + _mHD * 2 + 1];
		ptr_origin_objidx[_MLRCS * 2] = tmp_objidx[d_sorted_array[blockIdx.x] * OBJECTIVE_NUM * 2 + _MLRCS * 2];
		ptr_origin_objidx[_MLRCS * 2 + 1] = tmp_objidx[d_sorted_array[blockIdx.x] * OBJECTIVE_NUM * 2 + _MLRCS * 2 + 1];

		ptr_origin_lrcsval[P] = tmp_lrcsval[d_sorted_array[blockIdx.x] * 3 + P];
		ptr_origin_lrcsval[Q] = tmp_lrcsval[d_sorted_array[blockIdx.x] * 3 + Q];
		ptr_origin_lrcsval[L] = tmp_lrcsval[d_sorted_array[blockIdx.x] * 3 + L];
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
		if (threadIdx.x == 0)
		{
			do
			{
				*mutation_type = (char)(curand_uniform(&localState) * 4);
			} while (*mutation_type == 4);
		}
		__syncthreads();

		switch (*mutation_type)
		{
		case 0: // all random
			num_partition = ((c_len_amino_seq * c_cds_num) % blockDim.x == 0) ? (c_len_amino_seq * c_cds_num) / blockDim.x : (c_len_amino_seq * c_cds_num) / blockDim.x + 1;
			for (i = 0; i < num_partition; i++)
			{
				idx = blockDim.x * i + threadIdx.x;
				if (idx < c_len_amino_seq * c_cds_num)
				{
					seq_idx = idx % c_len_amino_seq;

					pos = FindNum_C(&c_codons[c_amino_startpos[s_amino_seq_idx[seq_idx]] * CODON_SIZE], &ptr_target_sol[idx * CODON_SIZE],
									c_codons_num[s_amino_seq_idx[seq_idx]]);
					mutation(&localState, &c_codons[c_amino_startpos[s_amino_seq_idx[seq_idx]] * CODON_SIZE], &ptr_target_sol[idx * CODON_SIZE],
							 c_codons_num[s_amino_seq_idx[seq_idx]], pos, c_mprob, RANDOM);
				}
			}
			break;

		case 1: // mCAI
			num_partition = (c_len_amino_seq % blockDim.x == 0) ? (c_len_amino_seq / blockDim.x) : (c_len_amino_seq / blockDim.x) + 1;
			for (i = 0; i < num_partition; i++)
			{
				seq_idx = blockDim.x * i + threadIdx.x;
				if (seq_idx < c_len_amino_seq)
				{
					pos = FindNum_C(&c_codons[c_amino_startpos[s_amino_seq_idx[seq_idx]] * CODON_SIZE],
									&ptr_target_sol[len_cds * ptr_origin_objidx[_mCAI * 2] + seq_idx * CODON_SIZE], c_codons_num[s_amino_seq_idx[seq_idx]]);
					mutation(&localState, &c_codons[c_amino_startpos[s_amino_seq_idx[seq_idx]] * CODON_SIZE],
							 &ptr_target_sol[len_cds * ptr_origin_objidx[_mCAI * 2] + seq_idx * CODON_SIZE], c_codons_num[s_amino_seq_idx[seq_idx]], pos, c_mprob, UPPER);
				}
			}
			break;

		case 2: // mHD
			num_partition = (c_len_amino_seq % blockDim.x == 0) ? (c_len_amino_seq / blockDim.x) : (c_len_amino_seq / blockDim.x) + 1;
			for (i = 0; i < num_partition; i++)
			{
				seq_idx = blockDim.x * i + threadIdx.x;
				if (seq_idx < c_len_amino_seq)
				{
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
		for (i = 0; i < c_cds_num; i++)
		{
			s_obj_compute[threadIdx.x] = 1;

			for (j = 0; j < num_partition; j++)
			{
				seq_idx = blockDim.x * j + threadIdx.x;
				if (seq_idx < c_len_amino_seq)
				{
					pos = FindNum_C(&c_codons[c_amino_startpos[s_amino_seq_idx[seq_idx]] * CODON_SIZE], &ptr_target_sol[len_cds * i + seq_idx * CODON_SIZE],
									c_codons_num[s_amino_seq_idx[seq_idx]]);
					s_obj_compute[threadIdx.x] *= (float)pow(c_codons_weight[c_amino_startpos[s_amino_seq_idx[seq_idx]] + pos], 1.0 / c_len_amino_seq);
				}
			}
			__syncthreads();

			j = blockDim.x / 2;
			while (true)
			{
				if (threadIdx.x < j)
				{
					s_obj_compute[threadIdx.x] *= s_obj_compute[threadIdx.x + j];
				}
				__syncthreads();

				if (j == 1)
					break;

				if ((j % 2 == 1) && (threadIdx.x == 0))
				{
					s_obj_compute[0] *= s_obj_compute[j - 1];
				}
				__syncthreads();

				j /= 2;
			}

			if (threadIdx.x == 0)
			{
				if (i == 0)
				{
					ptr_target_objval[_mCAI] = s_obj_compute[0];
					ptr_target_objidx[_mCAI * 2] = i;
				}
				else if (s_obj_compute[0] <= ptr_target_objval[_mCAI])
				{
					ptr_target_objval[_mCAI] = s_obj_compute[0];
					ptr_target_objidx[_mCAI * 2] = i;
				}
			}
			__syncthreads();
		}

		/* calculate mHD */
		num_partition = (len_cds % blockDim.x == 0) ? (len_cds / blockDim.x) : (len_cds / blockDim.x) + 1;
		for (i = 0; i < c_cds_num - 1; i++)
		{
			for (j = i + 1; j < c_cds_num; j++)
			{
				s_obj_compute[threadIdx.x] = 0;

				for (k = 0; k < num_partition; k++)
				{
					seq_idx = blockDim.x * k + threadIdx.x;

					if (seq_idx < len_cds && (ptr_target_sol[len_cds * i + seq_idx] != ptr_target_sol[len_cds * j + seq_idx]))
					{
						s_obj_compute[threadIdx.x] += 1;
					}
				}
				__syncthreads();

				k = blockDim.x / 2;
				while (true)
				{
					if (threadIdx.x < k)
					{
						s_obj_compute[threadIdx.x] += s_obj_compute[threadIdx.x + k];
					}
					__syncthreads();

					if (k == 1)
						break;

					if ((k % 2 == 1) && (threadIdx.x == 0))
					{
						s_obj_compute[0] += s_obj_compute[k - 1];
					}
					__syncthreads();

					k /= 2;
				}

				if (threadIdx.x == 0)
				{
					if (i == 0 && j == 1)
					{
						ptr_target_objval[_mHD] = s_obj_compute[0] / len_cds;
						ptr_target_objidx[_mHD * 2] = i;
						ptr_target_objidx[_mHD * 2 + 1] = j;
					}
					else if (s_obj_compute[0] / len_cds <= ptr_target_objval[_mHD])
					{
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
		for (i = 0; i < c_cds_num; i++)
		{
			for (j = i; j < c_cds_num; j++)
			{
				idx = threadIdx.x;

				if (i == j)
				{
					while (idx < 2 * len_cds + 1)
					{
						if (idx < len_cds + 1)
						{
							l = idx + 1;
							seq_idx = len_cds - l;

							for (k = 0; k < l; k++)
							{
								if (k == 0 || (seq_idx == -1))
									tmp_l = 0;
								else if (ptr_target_sol[len_cds * i + seq_idx + k] == ptr_target_sol[len_cds * j + k - 1])
								{
									tmp_l++;
									if (tmp_l >= lrcs_l)
									{
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
						else
						{
							l = 2 * len_cds + 1 - idx;
							seq_idx = len_cds - l;

							for (k = 0; k < l; k++)
							{
								if (k == 0)
									tmp_l = 0;
								else if (ptr_target_sol[len_cds * i + k - 1] == ptr_target_sol[len_cds * j + seq_idx + k])
								{
									tmp_l++;
									if (tmp_l >= lrcs_l)
									{
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
						if (idx < len_cds + 1)
						{
							l = idx + 1;
							seq_idx = len_cds - l;
							for (k = 0; k < l; k++)
							{
								if (k == 0)
									tmp_l = 0;
								else if (ptr_target_sol[len_cds * i + seq_idx + k] == ptr_target_sol[len_cds * j + k - 1])
								{
									tmp_l++;
									if (tmp_l >= lrcs_l)
									{
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
						else
						{
							l = 2 * len_cds + 1 - idx;
							seq_idx = len_cds - l;

							for (k = 0; k < l; k++)
							{
								if (k == 0)
									tmp_l = 0;
								else if (ptr_target_sol[len_cds * i + k - 1] == ptr_target_sol[len_cds * j + seq_idx + k])
								{
									tmp_l++;
									if (tmp_l >= lrcs_l)
									{
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
		while (true)
		{
			if (threadIdx.x < j && (s_obj_compute[threadIdx.x + j] > s_obj_compute[threadIdx.x]))
			{
				s_obj_compute[threadIdx.x] = s_obj_compute[threadIdx.x + j];
				s_lrcs_tid[threadIdx.x] = s_lrcs_tid[threadIdx.x + j];
			}
			__syncthreads();

			if (j == 1)
				break;

			if ((j % 2 == 1) && (threadIdx.x == 0))
			{
				if (s_obj_compute[j - 1] > s_obj_compute[0])
				{
					s_obj_compute[0] = s_obj_compute[j - 1];
					s_lrcs_tid[0] = s_lrcs_tid[j - 1];
				}
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

		if (sol_num == FIRST_SOL)
		{
			ptr_origin_sol = s_sol1;
			ptr_origin_objval = s_sol1_objval;
			ptr_origin_objidx = s_sol1_objidx;
			ptr_origin_lrcsval = s_sol1_lrcsval;
			ptr_target_sol = s_sol2;
			ptr_target_objval = s_sol2_objval;
			ptr_target_objidx = s_sol2_objidx;
			ptr_target_lrcsval = s_sol2_lrcsval;
		}
		else
		{
			ptr_origin_sol = s_sol2;
			ptr_origin_objval = s_sol2_objval;
			ptr_origin_objidx = s_sol2_objidx;
			ptr_origin_lrcsval = s_sol2_lrcsval;
			ptr_target_sol = s_sol1;
			ptr_target_objval = s_sol1_objval;
			ptr_target_objidx = s_sol1_objidx;
			ptr_target_lrcsval = s_sol1_lrcsval;
		}
		__syncthreads();
	}

	/* copy from shared memory to global memory */
	num_partition = (len_sol % blockDim.x == 0) ? (len_sol / blockDim.x) : (len_sol / blockDim.x) + 1;
	for (i = 0; i < num_partition; i++)
	{
		idx = blockDim.x * i + threadIdx.x;
		if (idx < len_sol)
		{
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

	state[id] = localState; // update state of random number generator

	return;
}

int main()
{
	srand((unsigned int)time(NULL));

	/* To get information of Deivce */
	int dev = 0; // number of device (GPU)
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
	char *amino_seq;		// store amino sequences from input file
	char *h_amino_seq_idx;	// notify index of amino abbreviation array corresponding input amino sequences
	char *h_amino_startpos; // notify position of according amino abbreviation index
	char *h_pop;			// store population (a set of solutions)
	float *h_objval;		// store objective values of population (solution 1, solution 2 .... solution n)
	int len_amino_seq, len_cds, len_sol;
	int pop_size;
	int twice_pop;
	int total_cycle;
	int sorting_cycle;
	int cds_num; // size of solution equal to number of CDSs(codon sequences) in a solution
	float mprob; // mutation probability
	float min_dist;

	float lowest_mcai;						// for divide initial solution section
	int limit;

	char tmp;
	int i, j, k;
	int x;
	int idx;
	char buf[256];
	FILE *fp;

	int numBlocks;
	int threadsPerBlock;

	curandStateXORWOW *genState;
	cudaEvent_t d_start, d_end;
	char *d_amino_seq_idx;
	char *d_pop;
	float *d_objval;
	char *d_objidx;
	int *d_lrcsval;
	int *d_sorted_array;
	bool *d_F_set, *d_Sp_set;
	int *d_np, *d_rank_count;
	Sol *d_sol_struct;

	char *tmp_pop;
	float *tmp_objval;
	char *tmp_objidx;
	int *tmp_lrcsval;

	float kernel_time;
	float total_time = 0;

	/* input parameter values */
	printf("Input file name : ");
	scanf("%s", input_file);
	printf("Input number of cycle : ");
	scanf("%d", &total_cycle);
	if (total_cycle < 0)
	{
		printf("Input max cycle value >= 0\n");
		return EXIT_FAILURE;
	}
	printf("Input number of sorting cycle : ");
	scanf("%d", &sorting_cycle);
	if (sorting_cycle <= 0)
	{
		printf("Input sorting cycle value > 0\n");
		return EXIT_FAILURE;
	}
	printf("Input number of solution : ");
	scanf("%d", &pop_size);
	if (pop_size <= 0)
	{
		printf("Input number of solution > 0\n");
		return EXIT_FAILURE;
	}
	printf("Input number of CDSs in a solution : ");
	scanf("%d", &cds_num);
	if (cds_num <= 1)
	{
		printf("Input number of CDSs > 1\n");
		return EXIT_FAILURE;
	}
	printf("Input mutation probability (0 ~ 1 value) : ");
	scanf("%f", &mprob);
	if (mprob < 0 || mprob > 1)
	{
		printf("Input mutation probability (0 ~ 1 value) : \n");
		return EXIT_FAILURE;
	}
	printf("input number of limit : "); scanf("%d", &limit);
	printf("Input thread per block x value --> number of thread  warp size (32) * x : ");
	scanf("%d", &x);
	printf("For sorting kernel use number of twice_pop threads per block");

	/* read input file (fasta format) */
	fp = fopen(input_file, "r");
	if (fp == NULL)
	{
		printf("Line : %d Opening input file is failed", __LINE__);
		return EXIT_FAILURE;
	}

	numBlocks = pop_size;
	threadsPerBlock = WARP_SIZE * x;
	twice_pop = pop_size * 2;

	fseek(fp, 0, SEEK_END);
	len_amino_seq = ftell(fp);
	fseek(fp, 0, SEEK_SET);
	fgets(buf, 256, fp);
	len_amino_seq -= ftell(fp);

	amino_seq = (char *)malloc(sizeof(char) * len_amino_seq);

	idx = 0;
	while (!feof(fp))
	{
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

	h_amino_seq_idx = (char *)malloc(sizeof(char) * len_amino_seq);
	for (i = 0; i < len_amino_seq; i++)
	{
		idx = FindAminoIndex(amino_seq[i]);
		if (idx == NOT_FOUND)
		{
			printf("FindAminoIndex function is failed... \n");
			return EXIT_FAILURE;
		}
		h_amino_seq_idx[i] = idx;
	}

	h_amino_startpos = (char *)malloc(sizeof(char) * 20);
	h_amino_startpos[0] = 0;
	for (i = 1; i < 20; i++)
	{
		h_amino_startpos[i] = h_amino_startpos[i - 1] + Codons_num[i - 1];
	}

/* caculate the smallest mCAI value */
	lowest_mcai = 1.f;
	for (i = 0; i < len_amino_seq; i++) {
		lowest_mcai *= (float)pow(Codons_weight[h_amino_startpos[h_amino_seq_idx[i]]], 1.0 / len_amino_seq);
	}


	/* host memory allocation */
	h_pop = (char *)malloc(sizeof(char) * twice_pop * len_sol);
	h_objval = (float *)malloc(sizeof(float) * twice_pop * OBJECTIVE_NUM);

	/* device memory allocation */
	CHECK_CUDA(cudaEventCreate(&d_start))
	CHECK_CUDA(cudaEventCreate(&d_end))
	CHECK_CUDA(cudaMalloc((void **)&genState, sizeof(curandStateXORWOW) * numBlocks * threadsPerBlock))
	CHECK_CUDA(cudaMalloc((void **)&d_amino_seq_idx, sizeof(char) * len_amino_seq))
	CHECK_CUDA(cudaMalloc((void **)&d_pop, sizeof(char) * twice_pop * len_sol))
	CHECK_CUDA(cudaMalloc((void **)&d_objval, sizeof(float) * twice_pop * OBJECTIVE_NUM))
	CHECK_CUDA(cudaMalloc((void **)&d_objidx, sizeof(char) * twice_pop * OBJECTIVE_NUM * 2))
	CHECK_CUDA(cudaMalloc((void **)&d_lrcsval, sizeof(int) * twice_pop * 3))
	CHECK_CUDA(cudaMalloc((void **)&d_sorted_array, sizeof(int) * twice_pop))
	CHECK_CUDA(cudaMalloc((void **)&d_F_set, sizeof(bool) * twice_pop * twice_pop))
	CHECK_CUDA(cudaMalloc((void **)&d_Sp_set, sizeof(bool) * twice_pop * twice_pop))
	CHECK_CUDA(cudaMalloc((void **)&d_rank_count, sizeof(int) * twice_pop))
	CHECK_CUDA(cudaMalloc((void **)&d_np, sizeof(int) * twice_pop))
	CHECK_CUDA(cudaMalloc((void **)&d_sol_struct, sizeof(Sol) * twice_pop))

	CHECK_CUDA(cudaMalloc((void **)&tmp_pop, sizeof(char) * twice_pop * len_sol))
	CHECK_CUDA(cudaMalloc((void **)&tmp_objval, sizeof(float) * twice_pop * OBJECTIVE_NUM))
	CHECK_CUDA(cudaMalloc((void **)&tmp_objidx, sizeof(char) * twice_pop * OBJECTIVE_NUM * 2))
	CHECK_CUDA(cudaMalloc((void **)&tmp_lrcsval, sizeof(int) * twice_pop * 3))

	/* memory copy host to device */
	CHECK_CUDA(cudaMemcpy(d_amino_seq_idx, h_amino_seq_idx, sizeof(char) * len_amino_seq, cudaMemcpyHostToDevice))
	CHECK_CUDA(cudaMemcpyToSymbol(c_codons_weight, Codons_weight, sizeof(Codons_weight)))
	CHECK_CUDA(cudaMemcpyToSymbol(c_amino_startpos, h_amino_startpos, sizeof(char) * 20))
	CHECK_CUDA(cudaMemcpyToSymbol(c_codons, Codons, sizeof(Codons)))
	CHECK_CUDA(cudaMemcpyToSymbol(c_codons_num, Codons_num, sizeof(Codons_num)))
	CHECK_CUDA(cudaMemcpyToSymbol(c_len_amino_seq, &len_amino_seq, sizeof(int)))
	CHECK_CUDA(cudaMemcpyToSymbol(c_cds_num, &cds_num, sizeof(int)))
	CHECK_CUDA(cudaMemcpyToSymbol(c_mprob, &mprob, sizeof(float)))
	CHECK_CUDA(cudaMemcpyToSymbol(c_sort_popsize, &twice_pop, sizeof(int)))

	/* ------------------------------------------------ kerenl call ----------------------------------------------- */
	/* curand generator setting */
	CHECK_CUDA(cudaEventRecord(d_start))
	setup_kernel<<<numBlocks, threadsPerBlock>>>(genState, (unsigned long long)rand());
	CHECK_CUDA(cudaEventRecord(d_end))
	CHECK_CUDA(cudaEventSynchronize(d_end))
	CHECK_CUDA(cudaEventElapsedTime(&kernel_time, d_start, d_end))
	total_time += kernel_time / 1000.f;

	/* initialize solution */
	CHECK_CUDA(cudaEventRecord(d_start))
	GenSolution<<<numBlocks, threadsPerBlock, sizeof(int) * (threadsPerBlock + 3) + sizeof(float) * (threadsPerBlock + OBJECTIVE_NUM) + sizeof(char) * (len_sol + len_amino_seq + 2 * OBJECTIVE_NUM)>>>(genState, d_amino_seq_idx, d_pop, d_objval, d_objidx, d_lrcsval, d_sorted_array, limit, lowest_mcai);
	CHECK_CUDA(cudaEventRecord(d_end))
	CHECK_CUDA(cudaEventSynchronize(d_end))
	CHECK_CUDA(cudaEventElapsedTime(&kernel_time, d_start, d_end))
	total_time += kernel_time / 1000.f;


	void *args[] = {&d_sorted_array, &d_F_set, &d_Sp_set, &d_pop, &d_objval, &d_objidx, &d_lrcsval};
	j = (total_cycle % sorting_cycle == 0) ? total_cycle / sorting_cycle : total_cycle / sorting_cycle + 1;
	CHECK_CUDA(cudaEventRecord(d_start))
	for (i = 0; i < j; i++)
	{
		CHECK_CUDA(cudaMemcpy(tmp_pop, d_pop, sizeof(char) * len_sol * twice_pop, cudaMemcpyDeviceToDevice))
		CHECK_CUDA(cudaMemcpy(tmp_objval, d_objval, sizeof(float) * OBJECTIVE_NUM * twice_pop, cudaMemcpyDeviceToDevice))
		CHECK_CUDA(cudaMemcpy(tmp_objidx, d_objidx, sizeof(char) * OBJECTIVE_NUM * 2 * twice_pop, cudaMemcpyDeviceToDevice))
		CHECK_CUDA(cudaMemcpy(tmp_lrcsval, d_lrcsval, sizeof(int) * 3 * twice_pop, cudaMemcpyDeviceToDevice))
		if (i == j - 1 && (total_cycle % sorting_cycle != 0))
		{
			mainKernel<<<numBlocks, threadsPerBlock, sizeof(int) * (threadsPerBlock + 3 * 2) + sizeof(float) * (threadsPerBlock + OBJECTIVE_NUM * 2) + sizeof(char) * (len_sol * 2 + len_amino_seq + OBJECTIVE_NUM * 2 * 2 + 1)>>>(genState, d_amino_seq_idx, d_pop, d_objval, d_objidx, d_lrcsval, total_cycle % sorting_cycle, tmp_pop, tmp_objval, tmp_objidx, tmp_lrcsval, d_sorted_array);
		}
		else
		{
			mainKernel<<<numBlocks, threadsPerBlock, sizeof(int) * (threadsPerBlock + 3 * 2) + sizeof(float) * (threadsPerBlock + OBJECTIVE_NUM * 2) + sizeof(char) * (len_sol * 2 + len_amino_seq + OBJECTIVE_NUM * 2 * 2 + 1)>>>(genState, d_amino_seq_idx, d_pop, d_objval, d_objidx, d_lrcsval, sorting_cycle, tmp_pop, tmp_objval, tmp_objidx, tmp_lrcsval, d_sorted_array);
		}
		CHECK_CUDA(cudaMemset(d_F_set,false,sizeof(bool)*twice_pop * twice_pop))
		CHECK_CUDA(cudaMemset(d_Sp_set,false,sizeof(bool)*twice_pop * twice_pop))
		CHECK_CUDA(cudaLaunchKernel((void *)FastSortSolution, 1, twice_pop, args, sizeof(Sol) * twice_pop + sizeof(int) * twice_pop * 2 + sizeof(float) * twice_pop * OBJECTIVE_NUM))
	}
	CHECK_CUDA(cudaEventRecord(d_end))
	CHECK_CUDA(cudaEventSynchronize(d_end))
	CHECK_CUDA(cudaEventElapsedTime(&kernel_time, d_start, d_end))
	total_time += kernel_time / 1000.f;

	/* memory copy device to host */
	CHECK_CUDA(cudaMemcpy(h_pop, d_pop, sizeof(char) * twice_pop * len_sol, cudaMemcpyDeviceToHost))
	CHECK_CUDA(cudaMemcpy(h_objval, d_objval, sizeof(float) * twice_pop * OBJECTIVE_NUM, cudaMemcpyDeviceToHost))

	// for compute hypervolume & minimum distance out process
	for (i = 0; i < twice_pop; i++)
	{
		h_objval[i * OBJECTIVE_NUM + _mHD] /= 0.4;
	}
	// print minimum distance to ideal point
	min_dist = MinEuclid(h_objval, twice_pop);
	printf("minimum distance to the ideal point : %f\n", min_dist);

	/* print solution */
	// for (i = 0; i < pop_size * 2; i++)
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
	// }

	/* print objective value */
	// printf("\n\n");
	// for (i = 0; i < pop_size * 2; i++)
	//{
	//	printf("%d solution\n", i + 1);
	//	printf("mCAI : %f mHD : %f MLRCS : %f\n", h_objval[i * OBJECTIVE_NUM + _mCAI], h_objval[i * OBJECTIVE_NUM + _mHD], h_objval[i * OBJECTIVE_NUM + _MLRCS]);
	//	printf("mCAI idx : %d mHD idx : %d %d MLRCS idx : %d %d\n", h_objidx[i * OBJECTIVE_NUM * 2 + _mCAI * 2],
	//		h_objidx[i * OBJECTIVE_NUM * 2 + _mHD * 2], h_objidx[i * OBJECTIVE_NUM * 2 + _mHD * 2 + 1],
	//		h_objidx[i * OBJECTIVE_NUM * 2 + _MLRCS * 2], h_objidx[i * OBJECTIVE_NUM * 2 + _MLRCS * 2 + 1]);
	//	printf("P : %d Q : %d L : %d\n", h_lrcsval[i * 3 + P], h_lrcsval[i * 3 + Q], h_lrcsval[i * 3 + L]);
	// }

	std::chrono::system_clock::time_point start = std::chrono::system_clock::now();
	fp = fopen("test.txt", "w");
	/* for computing hypervolume write file */
	for (i = 0; i < twice_pop; i++)
	{
		fprintf(fp, "%f %f %f\n", -h_objval[i * OBJECTIVE_NUM + _mCAI], -h_objval[i * OBJECTIVE_NUM + _mHD], h_objval[i * OBJECTIVE_NUM + _MLRCS]);
	}
	fclose(fp);
	std::chrono::duration<double> sec = std::chrono::system_clock::now() - start;
	total_time += static_cast<float>(sec.count());
	printf("\n\n total time : %f\n\n", total_time);

	/* free deivce memory */
	CHECK_CUDA(cudaEventDestroy(d_start))
	CHECK_CUDA(cudaEventDestroy(d_end))
	CHECK_CUDA(cudaFree(genState))
	CHECK_CUDA(cudaFree(d_amino_seq_idx))
	CHECK_CUDA(cudaFree(d_pop))
	CHECK_CUDA(cudaFree(d_objval))
	CHECK_CUDA(cudaFree(d_objidx))
	CHECK_CUDA(cudaFree(d_lrcsval))
	CHECK_CUDA(cudaFree(d_sorted_array))
	CHECK_CUDA(cudaFree(d_F_set))
	CHECK_CUDA(cudaFree(d_Sp_set))
	CHECK_CUDA(cudaFree(d_rank_count))
	CHECK_CUDA(cudaFree(d_np))
	CHECK_CUDA(cudaFree(d_sol_struct))
	CHECK_CUDA(cudaFree(tmp_pop))
	CHECK_CUDA(cudaFree(tmp_objval))
	CHECK_CUDA(cudaFree(tmp_objidx))
	CHECK_CUDA(cudaFree(tmp_lrcsval))

	/* free host memory */
	free(amino_seq);
	free(h_amino_seq_idx);
	free(h_amino_startpos);
	free(h_pop);
	free(h_objval);

	return EXIT_SUCCESS;
}