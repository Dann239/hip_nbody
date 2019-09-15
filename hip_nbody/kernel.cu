#include "cuda_runtime.h"

#ifndef __HIPCC__
#include "device_launch_parameters.h"
#include "cuda_runtime_api.h"
#include "cuda_occupancy.h"
#endif

#ifdef __INTELLISENSE__
void __syncthreads() {}
#endif

#include "kernel.h"
#include "properties.h"

#include <iostream>
#include <stdio.h>
using namespace std;

#define d3_0 double3({0.,0.,0.})

struct vec {
	double* v_gpu[3];
	double* v_cpu[3];
	long long validity;
	
	__device__ double3 get(int i) const {
		return double3({ 
			v_gpu[X][i],
			v_gpu[Y][i],
			v_gpu[Z][i] });
	}
	__device__ void set(int i, double3 p) {
		v_gpu[X][i] = p.x;
		v_gpu[Y][i] = p.y;
		v_gpu[Z][i] = p.z;
	}
	
	void invalidate() {
		for (int i = 0; i < 3; i++) {
			cudaMemcpyAsync(v_cpu[i], v_gpu[i], MEM_LEN, cudaMemcpyDeviceToHost);
		}
		validity = false;
	}
	void init() {
		for (int i = 0; i < 3; i++) {
			cudaMalloc(&v_gpu[i], MEM_LEN);
			cudaHostAlloc(&v_cpu[i], MEM_LEN, cudaHostAllocMapped);
		}
		validity = true;
	}
	void get(double** v) {
		if (!validity) {
			cudaDeviceSynchronize();
			for (int i = 0; i < 3; i++) {
				//swap(v[i], v_cpu[i]);
				cudaMemcpy(v[i], v_gpu[i], MEM_LEN, cudaMemcpyDeviceToHost);
			}
		}
		validity = true;
	}
	void set(double** v) {
		for (int i = 0; i < 3; i++)
			cudaMemcpyAsync(v_gpu[i], v[i], MEM_LEN, cudaMemcpyHostToDevice);
		validity = true;
	}
	void destroy() {
		for (int i = 0; i < 3; i++) {
			cudaFreeHost(v_cpu[i]);
			cudaFree(v_gpu[i]);
		}
	}
};

double* pos[3];
double* vel[3];

vec vec_pos, vec_vel;
static double* energy;
static double* _energy;
static properties* props;

double total_energy = 0;

void alloc() {
	for (int i = 0; i < 3; i++) {
		cudaHostAlloc(&pos[i], MEM_LEN, cudaHostAllocMapped);
		cudaHostAlloc(&vel[i], MEM_LEN, cudaHostAllocMapped);
	}

	vec_pos.init();
	vec_vel.init();

	cudaMalloc(&energy, MEM_LEN);
	cudaHostAlloc(&_energy, MEM_LEN, cudaHostAllocMapped);

	cudaMalloc(&props, ELEMS_NUM * sizeof(properties));
	static properties* _props = (properties*)malloc(ELEMS_NUM * sizeof(properties));
	for(int i = 0; i < ELEMS_NUM; i++) _props[i].set_properties(ELEMS_TYPES[i]);
	cudaMemcpy(props, _props, ELEMS_NUM * sizeof(properties), cudaMemcpyHostToDevice);
}
void dealloc() {
	for (int i = 0; i < 3; i++) {
		cudaFreeHost(pos[i]);
		cudaFreeHost(vel[i]);
	}
	vec_pos.destroy();
	vec_pos.destroy();
	cudaFree(energy);
	cudaFree(props);
	cudaFreeHost(_energy);
	cudaDeviceReset();
}

void pull_values() {
	vec_pos.get(pos);
	vec_vel.get(vel);
}
void push_values() {
	vec_pos.set(pos);
	vec_vel.set(vel);
}

void print_err(bool force) {
	cudaDeviceSynchronize();
	cudaError_t err = cudaGetLastError();
	if(err || force)
	cout << cudaGetErrorString(err) << endl;
}

__device__ double hypot2(double3 p) {
	return p.x * p.x + p.y * p.y + p.z * p.z;
}
__device__ double3 round(double3 a) {
	return { round(a.x),round(a.y),round(a.z) };
}

#ifndef __HCC__
__device__ double3& operator-= (double3& a, double3 b) {
	a.x -= b.x;
	a.y -= b.y;
	a.z -= b.z;
	return a;
}
__device__ double3& operator+=(double3& a, double3 b) {
	a.x += b.x;
	a.y += b.y;
	a.z += b.z;
	return a;
}
#endif

__device__ double3& operator*= (double3& a, double b) {
	a.x *= b;
	a.y *= b;
	a.z *= b;
	return a;
}
__device__ double3 operator- (double3 a, double3 b) {
	return { a.x - b.x, a.y - b.y, a.z - b.z };
}
__device__ double3 operator* (double b, double3 a) {
	return { a.x * b, a.y * b, a.z * b };
}
__device__ double3 operator+ (double3 a, double3 b) {
	return { a.x + b.x, a.y + b.y, a.z + b.z };
}
__device__ double3 operator& (double3 a, double3 b) {
	return { a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x };
}

__device__ bool operator== (double3 a, double3 b) {
	return a.x == b.x && a.y == b.y && a.z == b.z;
}

__device__ properties combine(properties p, properties _p) {
	p.EPSILON = sqrt(p.EPSILON * _p.EPSILON);
	p.SIGMA = (p.SIGMA + _p.SIGMA) / 2;
	p.Q = (p.Q * _p.Q) / sqrt(abs(p.Q * _p.Q));
	return p;
}

__device__ void get_a(double3& a_lj, double3& a_em, double3 p, double3 _p, double ss_ss) {
	double3 d = p - _p;
	d -= round(d);

	double d2 = hypot2(d);

#ifdef ENABLE_LJ
	double r2 = d2 * ss_ss,
		r_2 = 1. / r2,
		r_4 = r_2 * r_2,
		r_6 = r_4 * r_2,
		r_8 = r_4 * r_4,
		_2r_14__r_8 = (r_6 - .5) * r_8;
	a_lj += (_2r_14__r_8 * d);
#endif
	
#ifdef ENABLE_EM
	double d_2 = 1 / d2,
		d_1 = sqrt(d_2);
	a_em += d_2 * d_1 * d;
#endif 
}
__device__ void get_e(double& e_lj, double& e_em, double3 p, double3 _p, double ss_ss) {
	double3 d = p - _p;
	d -= round(d);
	double d2 = hypot2(d);

#ifdef ENABLE_LJ
	double r2 = d2 * ss_ss,
		r_2 = 1. / r2,
		r_4 = r_2 * r_2,
		r_6 = r_4 * r_2;
	e_lj += (r_6 - 1) * r_6;
#endif

#ifdef ENABLE_EM
	double d_1 = 1 / sqrt(d2);
	e_em += d_1;
#endif
}

__shared__ double3 _pos[];
#define GPU_PAIR_INTERACTION_WRAPPER(__COEFFS__, __INIT__, __BODY__, __POST__)	\
	int tid = threadIdx.x,														\
	bid = blockIdx.x,															\
	ind = bid * blockDim.x + tid;												\
																				\
	double3 p = 1. / SIZE * vec_pos.get(ind),									\
	v = vec_vel.get(ind);														\
																				\
	int props_ind = 0;															\
	while(bid / (double)GRID_SIZE > props[0].divisions[++props_ind]);			\
	properties __P = props[props_ind - 1];										\
																				\
	double lj_coeff[ELEMS_NUM];													\
	double em_coeff[ELEMS_NUM];													\
	double ss_ss[ELEMS_NUM];													\
	for(int i = 0; i < ELEMS_NUM; i++) {										\
		properties _P = combine(__P, props[i]);									\
		ss_ss[i] = (SIZE * SIZE) / (_P.SIGMA * _P.SIGMA);						\
		__COEFFS__																\
	}																			\
																				\
	props_ind = 0;																\
	for (int i = 0; i < gridDim.x; i++) {										\
		__syncthreads();														\
		_pos[tid] = 1. / SIZE * vec_pos.get(i * blockDim.x + tid);				\
		__syncthreads();														\
																				\
		if (i / (double)GRID_SIZE > __P.divisions[props_ind + 1])				\
			props_ind++;														\
																				\
		__INIT__																\
																				\
		for (int j = 0; j < blockDim.x; j++) {									\
			double3 _p = _pos[j];												\
			if (i != bid || j != tid) {											\
				__BODY__														\
			}																	\
		}																		\
		__POST__																\
	}																			\
																				\
	p *= SIZE;                									


__global__ void euler_gpu(vec vec_pos, vec vec_vel, properties* props) {
	double3 a_lj = d3_0;
	double3 a_em = d3_0;
	
	GPU_PAIR_INTERACTION_WRAPPER(
		lj_coeff[i] = 48. * _P.EPSILON * SIZE / _P.SIGMA / _P.SIGMA / _P.M;
		em_coeff[i] = 1. / (4. * PI * EPSILON0) * _P.Q * _P.Q / SIZE / SIZE / _P.M;
	,
		double3 da_lj = d3_0;
		double3 da_em = d3_0;
	,
		get_a(da_lj, da_em, p, _p, ss_ss[props_ind]);
	,
		a_lj += lj_coeff[props_ind] * da_lj;
		a_em += em_coeff[props_ind] * da_em;
	)

	double3 a = a_lj + a_em;

	v += TIME_STEP * a;
	p += TIME_STEP * v;

	vec_pos.set(ind, p);
	vec_vel.set(ind, v);

}
__global__ void energy_gpu (vec vec_pos, vec vec_vel, double* energy, properties* props) {
	double e_lj = 0;
	double e_em = 0;
	
	GPU_PAIR_INTERACTION_WRAPPER(
		lj_coeff[i] = 2. * _P.EPSILON;
		em_coeff[i] = 1. / (8. * PI * EPSILON0) * _P.Q * _P.Q / SIZE;
	,
		double de_lj = 0;
		double de_em = 0;
	,
		get_e(de_lj, de_em, p, _p, ss_ss[props_ind]);
	,
		e_lj += lj_coeff[props_ind] * de_lj;
		e_em += em_coeff[props_ind] * de_em;
	);
	
	double e_k = __P.M * hypot2(v) / 2.;

	energy[ind] = e_k + e_em + e_lj;
}

void euler_steps(int steps) {
#ifndef __INTELLISENSE__
	for(int i = 0; i < steps; i++)
		euler_gpu <<< GRID_SIZE, BLOCK_SIZE, BLOCK_SIZE * sizeof(double3) >>> (vec_pos, vec_vel, props);
#endif
	vec_pos.invalidate();
	vec_vel.invalidate();

	total_energy = 0;
	for (int i = 0; i < AMOUNT; i++) {
		total_energy += _energy[i];
	}

#ifndef __INTELLISENSE__
	energy_gpu <<< GRID_SIZE, BLOCK_SIZE, BLOCK_SIZE * sizeof(double3) >>> (vec_pos, vec_vel, energy, props);
#endif
	cudaMemcpyAsync(_energy, energy, MEM_LEN, cudaMemcpyDeviceToHost);
}

void force_energy_calc() {
#ifndef __INTELLISENSE__
	energy_gpu <<< GRID_SIZE, BLOCK_SIZE, BLOCK_SIZE * sizeof(double3) >>> (vec_pos, vec_vel, energy, props);
#endif
	cudaMemcpy(_energy, energy, MEM_LEN, cudaMemcpyDeviceToHost);
	total_energy = 0;
	for (int i = 0; i < AMOUNT; i++) {
		total_energy += _energy[i];
	}
}

void print_chars() {
	cudaDeviceProp chars;
	cudaFuncAttributes attr;
	int numBlocks;
	
	cudaGetDeviceProperties(&chars, 0);
	printf("Device:\n");
	printf("major: %d\n", chars.major);
	printf("minor: %d\n", chars.minor);
	printf("canMapHostMemory: %d\n", chars.canMapHostMemory);
	printf("multiProcessorCount: %d\n", chars.multiProcessorCount);
	printf("sharedMemPerBlock: %zu\n", chars.sharedMemPerBlock);
	printf("sharedMemPerMultiprocessor: %zu\n", chars.sharedMemPerMultiprocessor);
	printf("maxThreadsDim: %d\n", chars.maxThreadsDim[0]);
	printf("maxThreadsPerMultiProcessor: %d\n", chars.maxThreadsPerMultiProcessor);
	printf("regsPerBlock: %d\n", chars.regsPerBlock);
	printf("regsPerMultiprocessor: %d\n", chars.regsPerMultiprocessor);
#ifndef __HIPCC__
	printf("sharedMemPerMultiprocessor: %zu\n", chars.sharedMemPerMultiprocessor);
	printf("kernelExecTimeoutEnabled: %d\n", chars.kernelExecTimeoutEnabled);
	printf("warpSize: %d\n", chars.warpSize);
#endif

	cudaFuncGetAttributes(&attr, euler_gpu);
	cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocks, euler_gpu, BLOCK_SIZE, 0);
	printf("\neuler_gpu:\n");
	printf("binaryVersion: %d\n", attr.binaryVersion);
	printf("ptxVersion: %d\n", attr.ptxVersion);
	printf("maxThreadsPerBlock: %d\n", attr.maxThreadsPerBlock);
	printf("numRegs: %d\n", attr.numRegs);
	printf("localSizeBytes: %zu\n", attr.localSizeBytes);
	printf("sharedSizeBytes: %zu\n", attr.sharedSizeBytes);
	printf("BlockSize = %d; BlocksPerMP = %d; Occupancy = %f\n", BLOCK_SIZE, numBlocks, (double) (numBlocks * BLOCK_SIZE) / (chars.maxThreadsPerMultiProcessor));
	
	cudaFuncGetAttributes(&attr, energy_gpu);
	cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocks, energy_gpu, BLOCK_SIZE, 0);
	printf("\nenergy_gpu:\n");
	printf("binaryVersion: %d\n", attr.binaryVersion);
	printf("ptxVersion: %d\n", attr.ptxVersion);
	printf("maxThreadsPerBlock: %d\n", attr.maxThreadsPerBlock);
	printf("numRegs: %d\n", attr.numRegs);
	printf("localSizeBytes: %zu\n", attr.localSizeBytes);
	printf("sharedSizeBytes: %zu\n", attr.sharedSizeBytes);
	printf("BlockSize = %d; BlocksPerMP = %d; Occupancy = %f\n", BLOCK_SIZE, numBlocks, (double)(numBlocks * BLOCK_SIZE) / (chars.maxThreadsPerMultiProcessor));


	printf("\nBest BlockSize options:\n");
	double max = 0;
	for (int i = chars.regsPerBlock / attr.numRegs / 32 * 32; i > 0; i-=32) {
		cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocks, euler_gpu, i, i * sizeof(double3));
		double occ = (double)(numBlocks * i / chars.warpSize) / (chars.maxThreadsPerMultiProcessor / chars.warpSize);
		if (occ > max) {
			printf("BlockSize = %d; BlocksPerMP = %d; Occupancy = %f\n", i, numBlocks, occ);
			//max = occ;
		}
	}
	
}