#include "cuda_runtime.h"

#ifndef __HIPCC__
#include "device_launch_parameters.h"
#include "cuda_runtime_api.h"
#endif

#ifdef __INTELLISENSE__
void __syncthreads() {}
#endif

#include "kernel.h"
#include "properties.h"

#include <iostream>
using namespace std;

#define d3_0 double3({0.,0.,0.})

struct vec {
	double* v_gpu[3];
	double* v_cpu[3];
	bool validity;
	
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

vec vec_pos, vec_vel;
static double* energy;
static double* _energy;
static properties* props;

double total_energy = 0;

properties::properties(int block) {
	for (int i = 1; i <= ELEMS_NUM; i++)
		if (1. * block / GRID_SIZE <= ELEMS_DIVISIONS[i]) {
			set_properties(ELEMS_TYPES[i - 1]);
			return;
		}
	cout << "INVALID PROPERTIES OBJECT CREATED\n";
	return;
}

void alloc() {
	for (int i = 0; i < 3; i++) {
		cudaHostAlloc(&pos[i], MEM_LEN, cudaHostAllocMapped);
		cudaHostAlloc(&vel[i], MEM_LEN, cudaHostAllocMapped);
	}

	vec_pos.init();
	vec_vel.init();

	cudaMalloc(&energy, MEM_LEN);
	cudaHostAlloc(&_energy, MEM_LEN, cudaHostAllocMapped);

	cudaMalloc(&props, GRID_SIZE * sizeof(properties));
	static properties _props[GRID_SIZE];
	for (int i = 0; i < GRID_SIZE; i++) _props[i] = properties(i);
	cudaMemcpy(props, _props, GRID_SIZE * sizeof(properties), cudaMemcpyHostToDevice);
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

void print_chars() {
	cudaDeviceProp chars;
	cudaGetDeviceProperties(&chars,0);
	printf("major: %d\n", chars.major);
	printf("minor: %d\n", chars.minor);
	printf("canMapHostMemory: %d\n", chars.canMapHostMemory);
	printf("multiProcessorCount: %d\n", chars.multiProcessorCount);
	printf("sharedMemPerBlock: %zu\n", chars.sharedMemPerBlock);
	printf("maxThreadsDim: %d\n", chars.maxThreadsDim[0]);
	printf("totalGlobalMem: %zu\n", chars.totalGlobalMem);
	printf("regsPerBlock: %d\n", chars.regsPerBlock);
#ifndef __HIPCC__
    printf("sharedMemPerMultiprocessor: %zu\n", chars.sharedMemPerMultiprocessor);
	printf("kernelExecTimeoutEnabled: %d\n", chars.kernelExecTimeoutEnabled);
	printf("warpSize: %d\n", chars.warpSize);
#endif
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

#define GPU_PAIR_INTERACTION_WRAPPER(__INIT__, __BODY__, __POST__)							\
	int tid = threadIdx.x,																	\
	bid = blockIdx.x,																		\
	ind = bid * blockDim.x + tid;															\
																							\
	double3 p = 1. / SIZE * vec_pos.get(ind),												\
	v = vec_vel.get(ind);																	\
																							\
	properties P = props[bid];																\
	__shared__ double3 _pos[BLOCK_SIZE];													\
	for (int i = 0; i < gridDim.x; i++) {													\
		__syncthreads();																	\
		_pos[tid] = 1. / SIZE * vec_pos.get(i * blockDim.x + tid);							\
		__syncthreads();																	\
																							\
		properties _P = combine(P, props[i]);												\
		double ss_ss = (SIZE * SIZE) / (_P.SIGMA * _P.SIGMA);								\
		__INIT__																			\
		for (int j = 0; j < blockDim.x; j++) {												\
			double3 _p = _pos[j];															\
			if (i != bid || j != tid) {														\
				__BODY__																	\
			}																				\
		}																					\
		__POST__																			\
	}																						\
																							\
	p *= SIZE;                

__global__ void euler_gpu(vec vec_pos, vec vec_vel, properties* props) {
	double3 a_lj = d3_0;
	double3 a_em = d3_0;
	GPU_PAIR_INTERACTION_WRAPPER(
		double3 da_lj = d3_0;
		double3 da_em = d3_0;
	,
		get_a(da_lj, da_em, p, _p, ss_ss);
	,
		a_lj += 48. * _P.EPSILON * SIZE / _P.SIGMA / _P.SIGMA / _P.M * da_lj;
		a_em += 1. / (4. * PI * EPSILON0) * _P.Q * _P.Q / SIZE / SIZE / _P.M * da_em;
	);

	double3 a = a_lj + a_em;

	v += TIME_STEP * a;
	p += TIME_STEP * v;

	if (v == d3_0)
		printf("%d %d\n", blockIdx.x, threadIdx.x);

	vec_pos.set(ind, p);
	vec_vel.set(ind, v);
}
__global__ void energy_gpu (vec vec_pos, vec vec_vel, double* energy, properties* props) {
	double e_lj = 0;
	double e_em = 0;
	
	GPU_PAIR_INTERACTION_WRAPPER(
		double de_lj = 0;
		double de_em = 0;
	,
		get_e(de_lj, de_em, p, _p, ss_ss);
	,
		e_lj += 2. * _P.EPSILON * de_lj;
		e_em += 1. / (8. * PI * EPSILON0) * _P.Q * _P.Q / SIZE * de_em;
	);
	
	double e_k = P.M * hypot2(v) / 2.;

	energy[ind] = e_k + e_em + e_lj;
}

void euler_steps(int steps) {
#ifndef __INTELLISENSE__
	for(int i = 0; i < steps; i++)
		euler_gpu <<< GRID_SIZE, BLOCK_SIZE >>> (vec_pos, vec_vel, props);
#endif
	vec_pos.invalidate();
	vec_vel.invalidate();

	total_energy = 0;
	for (int i = 0; i < AMOUNT; i++) {
		total_energy += _energy[i];
	}

#ifndef __INTELLISENSE__
	energy_gpu <<< GRID_SIZE, BLOCK_SIZE >>> (vec_pos, vec_vel, energy, props);
#endif
	cudaMemcpyAsync(_energy, energy, MEM_LEN, cudaMemcpyDeviceToHost);
}

void force_energy_calc() {
#ifndef __INTELLISENSE__
	energy_gpu <<< GRID_SIZE, BLOCK_SIZE >>> (vec_pos, vec_vel, energy, props);
#endif
	cudaMemcpy(_energy, energy, MEM_LEN, cudaMemcpyDeviceToHost);
	total_energy = 0;
	for (int i = 0; i < AMOUNT; i++) {
		total_energy += _energy[i];
	}
}
