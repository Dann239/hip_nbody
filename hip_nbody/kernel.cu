#include "cuda_runtime.h"

#ifndef __HIPCC__
#include "device_launch_parameters.h"
#include "cuda_runtime_api.h"
#endif

#if defined __INTELLISENSE__
void __syncthreads() {}
#endif

#include "kernel.h"
#include "properties.h"

#include <iostream>
#include <stdio.h>
using namespace std;

#define d3_0 double3({0.,0.,0.})
cudaStream_t stream = cudaStreamDefault;

__host__ __device__ double3 extract(const double* const v[3], int i) {
	return double3({ v[X][i], v[Y][i], v[Z][i] });
}

class vec {
private:
	double* v_gpu_old[3];
	double* v_gpu_new[3];
	double* v_cpu[3];
	long long validity;
public:
	__device__ double3 get(int i) const {
		return extract(v_gpu_old, i);
	}
	__device__ void set(int i, double3 p) const {
		v_gpu_new[X][i] = p.x;
		v_gpu_new[Y][i] = p.y;
		v_gpu_new[Z][i] = p.z;
	}
	void gpu_copy() {
		for(int i = 0; i < 3; i++)
			cudaMemcpyAsync(v_gpu_old[i], v_gpu_new[i], MEM_LEN, cudaMemcpyDeviceToDevice, stream);
	}
	void invalidate() {
		for (int i = 0; i < 3; i++)
			cudaMemcpyAsync(v_cpu[i], v_gpu_new[i], MEM_LEN, cudaMemcpyDeviceToHost, stream);
		validity = false;
	}
	void init() {
		for (int i = 0; i < 3; i++) {
			cudaMalloc(&v_gpu_old[i], MEM_LEN);
			cudaMalloc(&v_gpu_new[i], MEM_LEN);
			cudaMallocHost(&v_cpu[i], MEM_LEN);
		}
		validity = true;
	}
	void get_all(double** v) {
		if (!validity) {
			cudaStreamSynchronize(stream);
			for (int i = 0; i < 3; i++)
				swap(v[i], v_cpu[i]);
		}
		validity = true;
	}
	void set_all(double** v) {
		for (int i = 0; i < 3; i++)
			cudaMemcpyAsync(v_gpu_old[i], v[i], MEM_LEN, cudaMemcpyHostToDevice, stream);
		validity = true;
	}
	void destroy() {
		for (int i = 0; i < 3; i++) {
			cudaFreeHost(v_cpu[i]);
			cudaFree(v_gpu_old[i]);
			cudaFree(v_gpu_new[i]);
		}
	}
};

double* pos[3];
double* vel[3];

vec vec_pos, vec_vel;
static double* energy;
static double* _energy;
static properties* props;

double potential_energy = 0;
double kinetic_energy = 0;
double temperature = 0;
double total_energy = 0;

void alloc() {
	cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);
	
	cudaStreamCreate(&stream);

	for (int i = 0; i < 3; i++) {
		cudaMallocHost(&pos[i], MEM_LEN);
		cudaMallocHost(&vel[i], MEM_LEN);
	}

	vec_pos.init();
	vec_vel.init();

	cudaMalloc(&energy, MEM_LEN);
	cudaMallocHost(&_energy, MEM_LEN);

	cudaMalloc(&props, ELEMS_NUM * sizeof(properties));
	static properties* _props = (properties*)malloc(ELEMS_NUM * sizeof(properties));
	for(int i = 0; i < ELEMS_NUM; i++) _props[i].set_properties(ELEMS_TYPES[i]);
	cudaMemcpy(props, _props, ELEMS_NUM * sizeof(properties), cudaMemcpyHostToDevice);
}
void dealloc() {
	cudaStreamDestroy(stream);
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
	vec_pos.get_all(pos);
	vec_vel.get_all(vel);
}
void push_values() {
	vec_pos.set_all(pos);
	vec_vel.set_all(vel);
}

void print_err(bool force) {
	if(force)
		cudaDeviceSynchronize();
	cudaError_t err = cudaGetLastError();
	if(err || force)
		cout << cudaGetErrorString(err) << endl;
}

__host__ __device__ bool invalid_elem(int block, properties p, int i) {
	return block / (double)GRID_SIZE < p.divisions[i];
}

__host__ __device__ int get_elem(int block, properties p) {
	for (int i = 1; i <= ELEMS_NUM; i++)
		if (invalid_elem(block, p, i))
			return i - 1;
	return ERROR;
}

properties get_properties(int num) {
	return properties(ELEMS_TYPES[get_elem(num / BLOCK_SIZE, properties(ERROR))]);
}

__host__ __device__ double hypot2(double3 p) {
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
	double d_2 = 1. / d2,
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
	e_lj += (r_6 - 1.) * r_6;
#endif

#ifdef ENABLE_EM
	double d_1 = 1. / sqrt(d2);
	e_em += d_1;
#endif
}

#define GPU_PAIR_INTERACTION_WRAPPER(__COEFFS__, __INIT__, __BODY__, _P0OST__)	\
	int tid = threadIdx.x,														\
	bid = blockIdx.x,															\
	ind = bid * blockDim.x + tid;												\
																				\
	double3 p = 1. / SIZE * vec_pos.get(ind),									\
	v = vec_vel.get(ind);														\
																				\
	properties _P0 = props[get_elem(bid, props[0])];							\
																				\
	double lj_coeff[ELEMS_NUM];													\
	double em_coeff[ELEMS_NUM];													\
	double ss_ss[ELEMS_NUM];													\
	for(int i = 0; i < ELEMS_NUM; i++) {										\
		properties _P = props[i];												\
		double epsilon = sqrt(_P.EPSILON * _P0.EPSILON);						\
		double sigma = (_P.SIGMA + _P0.SIGMA) / 2;								\
		ss_ss[i] = (SIZE * SIZE) / (sigma * sigma);								\
		__COEFFS__																\
	}																			\
	extern __shared__ double shm[];												\
	double* _posx = shm;														\
	double* _posy = &shm[BLOCK_SIZE];											\
	double* _posz = &shm[BLOCK_SIZE * 2];										\
	int props_ind = 0;															\
	for (int i = 0; i < GRID_SIZE; i++) {										\
																				\
		__syncthreads();														\
		double3 _pos = 1. / SIZE * vec_pos.get(i * BLOCK_SIZE + tid);			\
		_posx[tid] = _pos.x; _posy[tid] = _pos.y; _posz[tid] = _pos.z;			\
																				\
		if ( invalid_elem(i, _P0, props_ind ))									\
			props_ind++;														\
																				\
																				\
		__INIT__																\
																				\
		__syncthreads();														\
		for (int j = 0; j < BLOCK_SIZE; j++) {									\
			double3 _p = double3({_posx[j],_posy[j],_posz[j]});					\
			if (i != bid || j != tid) {											\
				__BODY__														\
			}																	\
		}																		\
		_P0OST__																\
	}																			\
																				\
	p *= SIZE;


__global__
void euler_gpu(const vec vec_pos, const vec vec_vel, properties* props) {
	double3 a_lj = d3_0;
	double3 a_em = d3_0;

	GPU_PAIR_INTERACTION_WRAPPER(
		lj_coeff[i] = 48. * epsilon * SIZE / sigma / sigma / _P0.M;
		em_coeff[i] = 1. / (4. * PI * EPSILON0) * _P0.Q * _P.Q / SIZE / SIZE / _P0.M;
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
__global__
void energy_gpu (const vec vec_pos, const vec vec_vel, double* energy, properties* props) {
	double e_lj = 0;
	double e_em = 0;

	GPU_PAIR_INTERACTION_WRAPPER(
		lj_coeff[i] = 2. * epsilon;
		em_coeff[i] = 1. / (8. * PI * EPSILON0) * _P0.Q * _P.Q / SIZE;
	,
		double de_lj = 0;
		double de_em = 0;
	,
		get_e(de_lj, de_em, p, _p, ss_ss[props_ind]);
	,
		e_lj += lj_coeff[props_ind] * de_lj;
		e_em += em_coeff[props_ind] * de_em;
	);

	double e_k = _P0.M * hypot2(v) / 2.;

	energy[ind] = e_em + e_lj;
}

void energy_calc() {
	potential_energy = 0;
	kinetic_energy = 0;
	for (int i = 0; i < AMOUNT; i++) {
		potential_energy += _energy[i] / AMOUNT;
		kinetic_energy += get_properties(i).M * hypot2(extract(vel, i)) / 2. / AMOUNT;
	}
	total_energy = potential_energy + kinetic_energy;
}

void euler_steps(int steps) {
	for(int i = 0; i < steps; i++) {
	#ifndef __INTELLISENSE__
		euler_gpu <<< GRID_SIZE, BLOCK_SIZE, sizeof(double) * BLOCK_SIZE * 3, stream >>> (vec_pos, vec_vel, props);
	#endif
		vec_pos.gpu_copy();
		vec_vel.gpu_copy();
	}
	vec_pos.invalidate();
	vec_vel.invalidate();

	energy_calc();

#ifndef __INTELLISENSE__
	energy_gpu <<< GRID_SIZE, BLOCK_SIZE, sizeof(double) * BLOCK_SIZE * 3, stream >>> (vec_pos, vec_vel, energy, props);
#endif
	cudaMemcpyAsync(_energy, energy, MEM_LEN, cudaMemcpyDeviceToHost, stream);

}
void force_energy_calc() {
#ifndef __INTELLISENSE__
	energy_gpu <<< GRID_SIZE, BLOCK_SIZE, sizeof(double) * BLOCK_SIZE * 3 >>> (vec_pos, vec_vel, energy, props);
#endif
	cudaMemcpy(_energy, energy, MEM_LEN, cudaMemcpyDeviceToHost);
	energy_calc();
}

void print_chars() {
	cudaDeviceProp chars;

	cudaGetDeviceProperties(&chars, 0);
	printf("Device:\n");
	printf("major: %d\n", chars.major);
	printf("minor: %d\n", chars.minor);
	printf("canMapHostMemory: %d\n", chars.canMapHostMemory);
	printf("multiProcessorCount: %d\n", chars.multiProcessorCount);
	printf("sharedMemPerBlock: %zu\n", chars.sharedMemPerBlock);
	printf("maxThreadsDim: %d\n", chars.maxThreadsDim[0]);
	printf("maxThreadsPerMultiProcessor: %d\n", chars.maxThreadsPerMultiProcessor);
	printf("regsPerBlock: %d\n\n", chars.regsPerBlock);

#ifndef __HIPCC__
	printf("singleToDoublePrecisionPerfRatio: %d\n", chars.singleToDoublePrecisionPerfRatio);
	printf("kernelExecTimeoutEnabled: %d\n", chars.kernelExecTimeoutEnabled);
	printf("regsPerMultiprocessor: %d\n", chars.regsPerMultiprocessor);
	printf("sharedMemPerMultiprocessor: %zu\n", chars.sharedMemPerMultiprocessor);
	printf("warpSize: %d\n\n", chars.warpSize);
#endif
	cudaFuncAttributes attr;
	cudaFuncGetAttributes(&attr, euler_gpu);
	printf("euler_gpu:\n");
	printf("binaryVersion: %d\n", attr.binaryVersion);
	printf("ptxVersion: %d\n", attr.ptxVersion);
	printf("maxThreadsPerBlock: %d\n", attr.maxThreadsPerBlock);
	printf("numRegs: %d\n", attr.numRegs);
	printf("localSizeBytes: %zu\n", attr.localSizeBytes);
	printf("sharedSizeBytes: %zu\n", attr.sharedSizeBytes);

#ifndef __HCC__
	int numBlocks;
	cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocks, (const void*)euler_gpu, BLOCK_SIZE, 0);
	printf("BlockSize = %d; BlocksPerMP = %d; Occupancy = %f\n", BLOCK_SIZE, numBlocks, (double) (numBlocks * BLOCK_SIZE) / (chars.maxThreadsPerMultiProcessor));
/*	printf("\nBest BlockSize options:\n");
	for (int i = 128; i <= 1024 ; i *= 2) {
		cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocks, (const void*)euler_gpu, i, 0);
		double occ = (double)(numBlocks * i) / (chars.maxThreadsPerMultiProcessor);

		printf("BlockSize = %d; BlocksPerMP = %d; Occupancy = %f\n", i, numBlocks, occ);
	}
*/
	printf("\n");
#endif
}
