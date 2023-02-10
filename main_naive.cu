/* include C/C++ header */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

/* include CUDA header */
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <curand_kernel.h>

#define WARP_SIZE 32

#define NOT_FOUND -1
#define MAX_CODON 6
#define CODON_SIZE 3
#define RANDOM 0
#define UPPER 1

#define OBJECTIVE_NUM 3
#define _mCAI 0
#define _mHD 1
#define _MLRCS 2

/* optional value */
//#define CODON_PER_THREAD 3					// number of codon one thread is responsible for a CDS in a solution (set of CDSs)

/* -------------------- 20 kinds of amino acids & weights are sorted ascending order -------------------- */
char Amino_abbreviation[20] = { 'A','C','D','E','F','G','H','I','K','L','M','N','P','Q','R','S','T','V','W','Y' };
char Codons[] = "GCGGCAGCCGCU\
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
float Codons_weight[] = { 1854 / 13563.0f, 5296 / 13563.0f, 7223 / 135063.0f, 1.0f,\
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


__device__ int findCodonNumIndx(const char* r_codons, const char* codon, const int r_num_codons)
{
	int i;
	int idx;

	for (i = 0; i < r_num_codons; i++)
	{
		if (*codon == r_codons[i * CODON_SIZE] && *(codon + 1) == r_codons[i * CODON_SIZE + 1] && *(codon + 2) == r_codons[i * CODON_SIZE + 2]) {
			idx = i;
			break;
		}
	}

	return idx;
}

__device__ void copyCodon(const char* origin, char* target)
{
	*target = *origin;
	*(target + 1) = *(origin + 1);
	*(target + 2) = *(origin + 2);

	return;
}

//__device__ void genSolution(curandStateXORWOW* state, char* s_codon, const char* r_codons, int r_num_codons, int type)
//{
//	int tmp;
//	float idx;
//	curandStateXORWOW localState = *state;
//
//	switch (type) {
//	
//	case RANDOM:
//		do {
//			idx = curand_uniform(&localState);
//			tmp = (int)(idx * r_num_codons);
//		} while (tmp == r_num_codons);
//		tmp *= CODON_SIZE;
//
//		*s_codon = r_codons[tmp];
//		*(s_codon + 1) = r_codons[tmp + 1];
//		*(s_codon + 2) = r_codons[tmp + 2];
//		break;
//
//	case UPPER:
//		*s_codon = r_codons[(r_num_codons - 1) * CODON_SIZE];
//		*(s_codon + 1) = r_codons[(r_num_codons - 1) * CODON_SIZE + 1];
//		*(s_codon + 2) = r_codons[(r_num_codons - 1) * CODON_SIZE + 2];
//		break;
//	}
//
//	*state = localState;
//
//	return;
//}

/* mutate codon upper adaptation or randmom adaptation */
__device__ void mutation(curandStateXORWOW* state, char* target_codon, const char* origin_codon, const char* r_codons, int r_num_codons, const float mprob, const int type)
{
	int i;
	int idx;
	int tmp;
	float cd_prob;
	float new_idx;
	curandStateXORWOW localState = *state;

	idx = findCodonNumIndx(r_codons, origin_codon, r_num_codons);

	/* 1.0 is included and 0.0 is excluded */
	cd_prob = curand_uniform(&localState);
	new_idx = curand_uniform(&localState);

	tmp = (int)(new_idx * r_num_codons);
	switch (type)
	{
	case RANDOM:
		if (cd_prob <= mprob && r_num_codons > 1) {
			while (idx == tmp || tmp == r_num_codons) {
				new_idx = curand_uniform(&localState);
				tmp = (int)(new_idx * r_num_codons);
			}
			*target_codon = r_codons[tmp * CODON_SIZE];
			*(target_codon + 1) = r_codons[tmp * CODON_SIZE + 1];
			*(target_codon + 2) = r_codons[tmp * CODON_SIZE + 2];
		}
		break;

	case UPPER:
		if (idx == r_num_codons - 1)
			break;
		if (cd_prob <= mprob) {
			while (idx > tmp) {
				new_idx = curand_uniform(&localState);
				tmp = (int)(new_idx * r_num_codons);
			}
			*target_codon = r_codons[tmp * CODON_SIZE];
			*(target_codon + 1) = r_codons[tmp * CODON_SIZE + 1];
			*(target_codon + 2) = r_codons[tmp * CODON_SIZE + 2];
		}
		break;
	}

	*state = localState;

	return;
}

/* For calculate 3 objective values */
__device__ bool mCAI(const float* s_cai, float* m_cai, int size)
{
	int i;
	float res;

	res = 1;
	for (i = 0; i < size; i++)
	{
		res *= s_cai[i];
	}

	if (res <= *m_cai) {
		*m_cai = res;
		return true;			// mCAI value is updated
	}

	return false;				// not update
}

__device__ bool mHD(const int* s_hd, float* m_hd, int size, int len_cds)
{
	int i;
	int tmp;
	float res;

	tmp = 0;
	for (i = 0; i < size; i++)
	{
		tmp += s_hd[i];
	}
	res = (float)tmp / len_cds;


	if (res <= *m_hd) {
		*m_hd = res;
		return true;		// updated
	}

	return false;			// not updated
}

//__device__ void MLRCS(const char* s_cds, int* d_matrix, const int len_cds, const int num_cds, const int tid, int* p, int* q, int* len, int* obj_idx)
//{
//	int i, j, k, l;
//	int cover_size;
//
//	cover_size = CODON_PER_THREAD * 3;
//
//	*len = 0;
//	for (i = 0; i < num_cds; i++) {
//		for (j = i; j < num_cds; j++) {
//			for (k = 0; k < len_cds; k++) {
//				if (i == j) {
//					for (l = 0; l < cover_size; l++) {
//						if (cover_size * tid + l < len_cds) {
//							if (k == (cover_size * tid + l))
//								d_matrix[len_cds * k + cover_size * tid + l] = 0;
//							else if (s_cds[len_cds * i + cover_size * tid + l] == s_cds[len_cds * j + k]) {
//								if (k == 0 || (cover_size * tid + l) == 0)
//									d_matrix[len_cds * k + cover_size * tid + l] = 1;
//								else
//									d_matrix[len_cds * k + cover_size * tid + l] = d_matrix[len_cds * (k - 1) + cover_size * tid + l - 1] + 1;
//							}
//							else
//								d_matrix[len_cds * k + cover_size * tid + l] = 0;
//						}
//					}
//				}
//				else {
//					for (l = 0; l < cover_size; l++) {
//						if (cover_size * tid + l < len_cds) {
//							if (s_cds[len_cds * i + cover_size * tid + l] == s_cds[len_cds * j + k]) {
//								if (k == 0 || (cover_size * tid + l) == 0)
//									d_matrix[len_cds * k + cover_size * tid + l] = 1;
//								else
//									d_matrix[len_cds * k + cover_size * tid + l] = d_matrix[len_cds * (k - 1) + cover_size * tid + l - 1] + 1;
//							}
//							else
//								d_matrix[len_cds * k + cover_size * tid + l] = 0;
//						}
//					}
//				}
//
//				if (tid == 0) {
//					for (int m = 0; m < len_cds; m++) {
//						if (d_matrix[len_cds * k + m] >= *len) {
//							*len = d_matrix[len_cds * k + m];
//							*p = m - *len + 1;
//							*q = k - *len + 1;
//							obj_idx[0] = i;
//							obj_idx[1] = j;
//						}
//					}
//				}
//				__syncthreads();
//			}
//		}
//	}
//
//	return;
//}



/* curand generator state setting */
__global__ void setup_kernel(curandStateXORWOW* state, int seed)
{
	int id = threadIdx.x + blockIdx.x * blockDim.x;

	/* Each thread gets same seed, a different sequence number, no offset */
	curand_init(seed, id, 0, &state[id]);

	return;
}

__global__ void mainKernel(curandStateXORWOW* state, const char* d_codons, const char* d_codons_num, const float* d_codons_weight, const char* d_amino_seq_idx, const char * d_amino_startpos, 
	char* d_pop, float * d_objval, const int len_amino_seq, const int cds_num, const int cycle, const float mprob)
{
	int i, j, k;
	int idx, seq_idx;
	int pos;
	int id;
	int num_partition;
	int len_cds, len_sol;
	curandStateXORWOW localState;

	id = threadIdx.x + blockIdx.x * blockDim.x;
	localState = state[id];
	len_cds = len_amino_seq * CODON_SIZE;
	len_sol = len_cds * cds_num;


	/* -------------------- shared memory allocation -------------------- */
	extern __shared__ float smem[];
	/* read only */
	__shared__ char* s_amino_seq_idx;
	__shared__ char* s_amino_startpos;
	__shared__ char* s_codons;
	__shared__ char* s_codons_num;
	__shared__ float* s_codons_weight;
	/* read & write */
	__shared__ char* s_sol;							// original solution
	__shared__ char* s_msol;						// mutated solution
	__shared__ float* s_obj_compute;				// for computing mCAI & mHD value
	__shared__ float* s_sol_objval;
	__shared__ float* s_msol_objval;
	__shared__ int* s_sol_objidx;
	__shared__ int* s_msol_objidx;
	__shared__ int* mutation_type;

	s_amino_seq_idx = (char*)smem;
	s_amino_startpos = (char*)s_amino_seq_idx[len_amino_seq];
	s_codons = (char*)s_amino_startpos[20];
	s_codons_num = (char*)s_codons[183];
	s_codons_weight = (float*)s_codons_num[20];
	s_sol = (char*)&s_codons_weight[61];
	s_msol = (char*)&s_sol[len_sol];
	s_obj_compute = (float*)&s_msol[len_sol];
	s_sol_objval = (float*)&s_obj_compute[blockDim.x];
	s_msol_objval = (float*)&s_sol_objval[OBJECTIVE_NUM];
	s_sol_objidx = (int*)&s_msol_objval[OBJECTIVE_NUM * 2];
	s_msol_objidx = (int*)&s_sol_objidx[OBJECTIVE_NUM * 2];
	mutation_type = (int*)&s_msol_objidx[1];
	/* -------------------- end of shared memory allocation -------------------- */



	/* read only shared memory variable value setting */
	num_partition = (len_amino_seq % blockDim.x == 0) ? len_amino_seq / blockDim.x : len_amino_seq / blockDim.x + 1;
	for (i = 0; i < num_partition; i++) {
		idx = blockDim.x * i + threadIdx.x;
		if (idx < len_amino_seq)
			s_amino_seq_idx[idx] = d_amino_seq_idx[idx];
	}
	num_partition = 183 / blockDim.x + 1;
	for (i = 0; i < num_partition; i++) {
		idx = blockDim.x * i + threadIdx.x;
		if (idx < 183)
			s_codons[idx] = d_codons[idx];
	}
	num_partition = 61 / blockDim.x + 1;
	for (i = 0; i < num_partition; i++) {
		idx = blockDim.x * i + threadIdx.x;
		if (idx < 61)
			s_codons_weight[idx] = d_codons_weight[idx];
	}
	for (i = 0; i < 20; i++) {
		if (threadIdx.x < 20) {
			s_codons_num[threadIdx.x] = d_codons_num[threadIdx.x];
			s_amino_startpos[threadIdx.x] = d_amino_startpos[threadIdx.x];
		}
	}
	__syncthreads();


	/* -------------------- initialize solution -------------------- */
	num_partition = (len_amino_seq * cds_num % blockDim.x == 0) ? len_amino_seq * cds_num / blockDim.x : len_amino_seq * cds_num / blockDim.x + 1;
	for (i = 0; i < num_partition; i++) {
		idx = blockDim.x * i + threadIdx.x;
		if (idx < num_partition) {
			seq_idx = idx % len_amino_seq;
			
			do {
				pos = (int)(curand_uniform(&localState) * s_codons_num[s_amino_seq_idx[seq_idx]]);
			} while (pos == s_codons_num[s_amino_seq_idx[seq_idx]]);
			
			s_sol[idx * CODON_SIZE] = s_codons[(s_amino_startpos[s_amino_seq_idx[seq_idx]] + pos) * CODON_SIZE];
			s_sol[idx * CODON_SIZE + 1] = s_codons[(s_amino_startpos[s_amino_seq_idx[seq_idx]] + pos) * CODON_SIZE + 1];
			s_sol[idx * CODON_SIZE + 2] = s_codons[(s_amino_startpos[s_amino_seq_idx[seq_idx]] + pos) * CODON_SIZE + 2];
		}
	}
	/* -------------------- end of initialize -------------------- */



	/* calculate mCAI */
	if (threadIdx.x == 0)
		s_sol_objval[_mCAI] = 1;
	__syncthreads();

	num_partition = (len_amino_seq % blockDim.x == 0) ? len_amino_seq * cds_num / blockDim.x : len_amino_seq * cds_num / blockDim.x + 1;
	for (i = 0; i < cds_num; i++) {
		s_obj_compute[threadIdx.x] = 1;
		for (j = 0; j < CODON_PER_THREAD; j++) {
			if (threadIdx.x + blockDim.x * j < len_amino_seq) {
				idx = findCodonNumIndx(r_codons[j], &s_sol[len_cds * i + (threadIdx.x + blockDim.x * j) * CODON_SIZE], r_num_codons[j]);
				s_obj_compute[threadIdx.x] *= pow(r_weight[j][idx], 1.0 / len_amino_seq);
			}
		}
		__syncthreads();
		if (threadIdx.x == 0) {
			if (mCAI(s_obj_compute, &s_sol_objval[_mCAI], blockDim.x))
				s_sol_objidx[_mCAI * 2] = i;
		}
		__syncthreads();
	}


	/* calculate mHD */
	if (threadIdx.x == 0)
		s_sol_objval[_mHD] = 1;
	__syncthreads();

	for (i = 0; i < cds_num; i++) {
		for (j = i + 1; j < cds_num; j++) {
			s_hd[threadIdx.x] = 0;
			for (k = 0; k < CODON_PER_THREAD; k++) {
				if (threadIdx.x + blockDim.x * k < len_amino_seq) {
					for (int m = 0; m < CODON_SIZE; m++) {
						if (s_sol[len_cds * i + (threadIdx.x + blockDim.x * k) * CODON_SIZE + m] != s_sol[len_cds * j + (threadIdx.x + blockDim.x * k) * CODON_SIZE + m])
							s_hd[threadIdx.x]++;
					}
				}
			}
			__syncthreads();
			if (threadIdx.x == 0) {
				if (mHD(s_hd, &s_sol_objval[_mHD], blockDim.x, len_cds)) {
					s_sol_objidx[_mHD * 2] = i;
					s_sol_objidx[_mHD * 2 + 1] = j;
				}
			}
			__syncthreads();
		}
	}




	/* mutate cycle times */
	for (int c = 0; c < cycle; c++)
	{
		/* select mutatation type */
		if (threadIdx.x == 0) {
			do {
				*mutation_type = (int)(curand_uniform(&localState) * 3);
			} while (*mutation_type == 3);
		}
		__syncthreads();


		switch (*mutation_type) {
		case 0:			// all random
			for (i = 0; i < CODON_PER_THREAD; i++) {
				if (threadIdx.x + blockDim.x * i < len_amino_seq) {
					for (j = 0; j < cds_num; j++) {
						mutation(&state[id], &s_msol[len_cds * j + (threadIdx.x + blockDim.x * i) * CODON_SIZE], &s_sol[len_cds * j + (threadIdx.x + blockDim.x * i) * CODON_SIZE], r_codons[i], r_num_codons[i], mprob, RANDOM);
					}
				}
			}
			break;

		case 1:			// mCAI
			for (i = 0; i < CODON_PER_THREAD; i++) {
				if (threadIdx.x + blockDim.x * i < len_amino_seq) {
					mutation(&state[id], &s_msol[len_cds * s_sol_objidx[_mCAI * 2] + (threadIdx.x + blockDim.x * i) * CODON_SIZE], &s_sol[len_cds * s_sol_objidx[_mCAI * 2] + (threadIdx.x + blockDim.x * i) * CODON_SIZE],
						r_codons[i], r_num_codons[i], mprob, UPPER);
				}
			}
			break;

		case 2:			// mHD
			for (i = 0; i < CODON_PER_THREAD; i++) {
				if (threadIdx.x + blockDim.x * i < len_amino_seq) {
					mutation(&state[id], &s_msol[len_cds * s_sol_objidx[_mHD * 2] + (threadIdx.x + blockDim.x * i) * CODON_SIZE], &s_sol[len_cds * s_sol_objidx[_mHD * 2] + (threadIdx.x + blockDim.x * i) * CODON_SIZE],
						r_codons[i], r_num_codons[i], mprob, RANDOM);
					mutation(&state[id], &s_msol[len_cds * s_sol_objidx[_mHD * 2 + 1] + (threadIdx.x + blockDim.x * i) * CODON_SIZE], &s_sol[len_cds * s_sol_objidx[_mHD * 2 + 1] + (threadIdx.x + blockDim.x * i) * CODON_SIZE],
						r_codons[i], r_num_codons[i], mprob, RANDOM);
				}
			}
			break;
		}
		__syncthreads();



		/* caculate mCAI */
		if (threadIdx.x == 0)
			s_msol_objval[_mCAI] = 1;
		__syncthreads();

		for (i = 0; i < cds_num; i++) {
			s_obj_compute[threadIdx.x] = 1;
			for (j = 0; j < CODON_PER_THREAD; j++) {
				if (threadIdx.x + blockDim.x * j < len_amino_seq) {
					idx = findCodonNumIndx(r_codons[j], &s_msol[len_cds * i + (threadIdx.x + blockDim.x * j) * CODON_SIZE], r_num_codons[j]);
					s_obj_compute[threadIdx.x] *= pow(r_weight[j][idx], 1.0 / len_amino_seq);
				}
			}
			__syncthreads();
			if (threadIdx.x == 0) {
				if (mCAI(s_obj_compute, &s_msol_objval[_mCAI], blockDim.x))
					s_msol_objidx[_mCAI * 2] = i;
			}
			__syncthreads();
		}


		/* caculate mHD */
		if (threadIdx.x == 0)
			s_msol_objval[_mHD] = 1;
		__syncthreads();

		for (i = 0; i < cds_num; i++) {
			for (j = i + 1; j < cds_num; j++) {
				s_hd[threadIdx.x] = 0;
				for (k = 0; k < CODON_PER_THREAD; k++) {
					if (threadIdx.x + blockDim.x * k < len_amino_seq) {
						for (int m = 0; m < CODON_SIZE; m++) {
							if (s_msol[len_cds * i + (threadIdx.x + blockDim.x * k) * CODON_SIZE + m] != s_msol[len_cds * j + (threadIdx.x + blockDim.x * k) * CODON_SIZE + m])
								s_hd[threadIdx.x]++;
						}
					}
				}
				__syncthreads();
				if (threadIdx.x == 0) {
					if (mHD(s_hd, &s_msol_objval[_mHD], blockDim.x, len_cds)) {
						s_msol_objidx[_mHD * 2] = i;
						s_msol_objidx[_mHD * 2 + 1] = j;
					}
				}
				__syncthreads();
			}
		}


		/* compare & copy */
		if (threadIdx.x == 0) {
			if (s_msol_objval[_mCAI] >= s_sol_objval[_mCAI] &&
				s_msol_objval[_mHD] >= s_sol_objval[_mHD] &&
				s_msol_objval[_MLRCS] <= s_sol_objval[_MLRCS])
				*check = 1;
			else
				*check = 0;
		}
		__syncthreads();

		if (*check == 1) {
			for (i = 0; i < CODON_PER_THREAD; i++) {
				if (threadIdx.x + blockDim.x * i < len_amino_seq) {
					for (j = 0; j < cds_num; j++) {
						copyCodon(&s_msol[len_cds * j + (threadIdx.x + blockDim.x * i) * CODON_SIZE], &s_sol[len_cds * j + (threadIdx.x + blockDim.x * i) * CODON_SIZE]);
					}
				}
			}

			if (threadIdx.x == 0) {
				s_sol_objval[_mCAI] = s_msol_objval[_mCAI];
				s_sol_objval[_mHD] = s_msol_objval[_mHD];
				s_sol_objval[_MLRCS] = s_msol_objval[_MLRCS];
				s_sol_objidx[_mCAI * 2] = s_msol_objidx[_mCAI * 2];
				s_sol_objidx[_mHD * 2] = s_msol_objidx[_mHD * 2];
				s_sol_objidx[_mHD * 2 + 1] = s_msol_objidx[_mHD * 2 + 1];
				s_sol_objidx[_MLRCS * 2] = s_msol_objidx[_MLRCS * 2];
				s_sol_objidx[_MLRCS * 2 + 1] = s_msol_objidx[_MLRCS * 2 + 1];
				*p = *tmp_p;
				*q = *tmp_q;
				*l = *tmp_l;
			}
		}
		__syncthreads();

	}


	/* copy from shared memory to global memory */
	num_partition = (len_sol % blockDim.x == 0) ? len_sol / blockDim.x : len_sol / blockDim.x + 1;
	for (i = 0; i < num_partition; i++) {
		if (blockDim.x * i + threadIdx.x < len_sol)
			d_pop[blockIdx.x * len_sol + blockDim.x * i + threadIdx.x] = s_sol[blockDim.x * i + threadIdx.x];
	}

	if (threadIdx.x == 0)
	{
		d_objval[blockIdx.x * OBJECTIVE_NUM + _mCAI] = s_sol_objval[_mCAI];
		d_objval[blockIdx.x * OBJECTIVE_NUM + _mHD] = s_sol_objval[_mHD];
	}

	return;
}



int main()
{
	srand(time(NULL));

	char input_file[32] = "Q5VZP5.fasta.txt";
	char* amino_seq;						// store amino sequences from input file
	char* h_amino_seq_idx;					// notify index of amino abbreviation array corresponding input amino sequences
	char* h_pop;							// store population (a set of solutions)
	float* h_objval;						// store objective values of population (solution 1, solution 2 .... solution n)
	char* h_amino_startpos;					// notify position of according amino abbreviation index
	int len_amino_seq, len_cds, len_sol;
	int pop_size;
	int cycle;
	int cds_num;							// size of solution equal to number of CDSs(codon sequences) in a solution
	float mprob;							// mutation probability
	float lowest_mcai;						// for divide initial solution section
	
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
	char* d_amino_startpos;
	char* d_codons;
	char* d_codons_num;
	float* d_codons_weight;
	curandStateXORWOW* genState;

	/* for time and mcai section cehck */
	cudaEvent_t d_start, d_end;
	float kernel_time;
	cudaEventCreate(&d_start);
	cudaEventCreate(&d_end);



	/* ---------------------------------------- preprocessing ---------------------------------------- */
	/* input parameter values */
	//printf("input file name : "); scanf_s("%s", &input_file);
	printf("input number of cycle : "); scanf_s("%d", &cycle);					// if number of cycle is zero we can check initial population
	if (cycle < 0) {
		printf("input max cycle value >= 0\n");
		return EXIT_FAILURE;
	}
	printf("input number of solution : "); scanf_s("%d", &pop_size);
	if (pop_size <= 0) {
		printf("input number of solution > 0\n");
		return EXIT_FAILURE;
	}
	printf("input number of CDSs in a solution : "); scanf_s("%d", &cds_num);
	if (cds_num <= 1) {
		printf("input number of CDSs > 1\n");
		return EXIT_FAILURE;
	}
	printf("input mutation probability (0 ~ 1 value) : "); scanf_s("%f", &mprob);
	if (mprob < 0 || mprob > 1) {
		printf("input mutation probability (0 ~ 1 value) : \n");
		return EXIT_FAILURE;
	}


	/* read input file (fasta format) */
	fopen_s(&fp, input_file, "r");
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

	/* caculate the smallest mCAI value */
	lowest_mcai = 1.f;
	for (i = 0; i < len_amino_seq; i++) {
		lowest_mcai *= pow(Codons_weight[h_amino_startpos[i]], 1.0 / len_amino_seq);
	}
	/* ---------------------------------------- end of preprocessing ---------------------------------------- */


	x = 1;		// optional
	threadsPerBlock = WARP_SIZE * x;
	numBlocks = pop_size;

	/* host memory allocation */
	h_pop = (char*)malloc(sizeof(char) * pop_size * len_sol);
	h_objval = (float*)malloc(sizeof(float) * pop_size * OBJECTIVE_NUM);


	/* device memory allocation */
	cudaMalloc((void**)&genState, sizeof(curandStateXORWOW) * numBlocks * threadsPerBlock);
	cudaMalloc((void**)&d_codons, sizeof(Codons));
	cudaMalloc((void**)&d_codons_num, sizeof(Codons_num));
	cudaMalloc((void**)&d_codons_weight, sizeof(Codons_weight));
	cudaMalloc((void**)&d_amino_seq_idx, sizeof(char) * len_amino_seq);
	cudaMalloc((void**)&d_amino_startpos, sizeof(char) * 20);
	cudaMalloc((void**)&d_pop, sizeof(char) * numBlocks * len_sol);
	cudaMalloc((void**)&d_objval, sizeof(float) * numBlocks * OBJECTIVE_NUM);


	/* memory copy host to device */
	cudaMemcpy(d_pop, h_pop, sizeof(char) * numBlocks * len_sol, cudaMemcpyHostToDevice);
	cudaMemcpy(d_amino_seq_idx, h_amino_seq_idx, sizeof(char) * len_amino_seq, cudaMemcpyHostToDevice);
	cudaMemcpy(d_amino_startpos, h_amino_startpos, sizeof(char) * 20, cudaMemcpyHostToDevice);
	cudaMemcpy(d_codons, Codons, sizeof(Codons), cudaMemcpyHostToDevice);
	cudaMemcpy(d_codons_num, Codons_num, sizeof(Codons_num), cudaMemcpyHostToDevice);
	cudaMemcpy(d_codons_weight, Codons_weight, sizeof(Codons_weight), cudaMemcpyHostToDevice);


	/* optimize kerenl call */
	setup_kernel << <numBlocks, threadsPerBlock >> > (genState, rand());

	cudaEventRecord(d_start);
	//mainKernel << <numBlocks, threadsPerBlock, sizeof(char)* len_sol * 2 + sizeof(int) * (8 + threadsPerBlock + OBJECTIVE_NUM * 4) + sizeof(float) * (threadsPerBlock + OBJECTIVE_NUM * 2) >> >
	//	(genState, d_pop, d_objval, d_amino_seq_idx, d_start_idx, d_codons, d_num_codons, d_weight, mprob, max_cycle, num_cds, len_amino_seq, d_matrix, d_check_mcai);
	cudaEventRecord(d_end);
	cudaEventSynchronize(d_end);
	cudaEventElapsedTime(&kernel_time, d_start, d_end);


	/* memory copy device to host */
	cudaMemcpy(h_pop, d_pop, sizeof(char) * numBlocks * len_sol, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_objval, d_objval, sizeof(float) * numBlocks * OBJECTIVE_NUM, cudaMemcpyDeviceToHost);



	/* print solution */
	//for (i = 0; i < pop_size; i++)
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
	//
	///* print objective value */
	//for (i = 0; i < pop_size; i++)
	//{
	//	printf("%d solution\n", i + 1);
	//	printf("mCAI : %f mHD : %f\n", h_objval[i * OBJECTIVE_NUM + _mCAI], h_objval[i * OBJECTIVE_NUM + _mHD]);
	//}


	/* for computing hypervolume write file */
	//fopen_s(&fp, "test.txt", "w");
	//for (i = 0; i < pop_size; i++)
	//{
	//	fprintf(fp, "%f %f %f\n", -h_objval[i * OBJECTIVE_NUM + _mCAI], -h_objval[i * OBJECTIVE_NUM + _mHD] / 0.35, h_objval[i * OBJECTIVE_NUM + _MLRCS]);
	//}
	//fclose(fp);



	printf("\nGPU kerenl cycle time : %f second\n",  kernel_time/ 1000.f);



	///* check mCAI vlaue section count chekck */
	//int check_cnt[10];
	//float check_low, check_high;
	//memset(check_cnt, 0, sizeof(int) * 10);
	//for (i = 0; i < pop_size; i++) {
	//	check_low = 0;
	//	check_high = 0.1f;
	//	j = 0;
	//	while (true) {
	//		if (check_low < h_check_mcai[i] && h_check_mcai[i] <= check_high) {
	//			check_cnt[j] += 1;
	//			break;
	//		}
	//		else {
	//			check_low += 0.1f;
	//			check_high += 0.1f;
	//			j++;
	//		}
	//	}
	//}
	//check_low = 0;
	//check_high = 0.1f;
	//for (i = 0; i < 10; i++) {
	//	printf("mCAI %f <   <= %f    counting : %d\n", check_low, check_high, check_cnt[i]);
	//	check_low += 0.1f;
	//	check_high += 0.1f;
	//}
	///* end of check */



	/* free deivce memory */
	cudaFree(genState);
	cudaFree(d_codons);
	cudaFree(d_codons_num);
	cudaFree(d_codons_weight);
	cudaFree(d_amino_seq_idx);
	cudaFree(d_amino_startpos);
	cudaFree(d_pop);
	cudaFree(d_objval);

	cudaEventDestroy(d_start);
	cudaEventDestroy(d_end);

	/* free host memory */
	free(amino_seq);
	free(h_amino_seq_idx);
	free(h_amino_startpos);
	free(h_pop);
	free(h_objval);


	return EXIT_SUCCESS;
}