#include "cuda_runtime.h"

#ifndef __HIPCC__
#include "device_launch_parameters.h"
#include "cuda_runtime_api.h"
#endif

#if defined __INTELLISENSE__
void __syncthreads() {}
#include "math_functions.h"
#endif

#include "kernel.h"
#include "properties.h"

#include <iostream>
#include <stdio.h>
using namespace std;

#define d3_0 double3({0.,0.,0.})
cudaStream_t stream = cudaStreamDefault;

__host__ __device__ double3 extract(const double* const *v, int i, int offset = 0) {
	return double3({ v[offset + X][i], v[offset + Y][i], v[offset + Z][i] });
}

template<int size>
class vec {
private:
	double* v_gpu_old[size];
	double* v_gpu_new[size];
	double* v_cpu[size];
	long long validity;
public:
	double* v_raw[size];

	__device__ double3 get(int i, int offset) const {
		return extract(v_gpu_old, i, offset);
	}
	__device__ void set(int i, double3 p, int offset) const {
		v_gpu_new[offset + X][i] = p.x;
		v_gpu_new[offset + Y][i] = p.y;
		v_gpu_new[offset + Z][i] = p.z;
	}
	__device__ void get_single(int i, int offset) const {
		return v_gpu_old[offset][i];
	}
	__device__ void set_single(int i, double p, int offset) const {
		v_gpu_new[offset][i] = p;
	}
	void gpu_copy() {
		cudaMemcpyAsync(v_gpu_old[0], v_gpu_new[0], MEM_LEN * size, cudaMemcpyDeviceToDevice, stream);
	}
	void invalidate() {
		cudaMemcpyAsync(v_cpu[0], v_gpu_new[0], MEM_LEN * size, cudaMemcpyDeviceToHost, stream);
		validity = false;
	}
	void init() {
		cudaMalloc(&v_gpu_old[0], MEM_LEN * size);
		cudaMalloc(&v_gpu_new[0], MEM_LEN * size);
		cudaMallocHost(&v_cpu[0], MEM_LEN * size);
		cudaMallocHost(&v_raw[0], MEM_LEN * size);
		for (int i = 1; i < size; i++) {
			v_gpu_old[i] = &v_gpu_old[0][i * AMOUNT];
			v_gpu_new[i] = &v_gpu_new[0][i * AMOUNT];
			v_cpu[i] = &v_cpu[0][i * AMOUNT];
			v_raw[i] = &v_raw[0][i * AMOUNT];
		}
		validity = true;
	}
	void get_all() {
		if (!validity) {
			cudaStreamSynchronize(stream);
			for (int i = 0; i < size; i++)
				swap(v_raw[i], v_cpu[i]);
		}
		validity = true;
	}
	void set_all() {
		cudaMemcpyAsync(v_gpu_old[0], v_raw[0], MEM_LEN * size, cudaMemcpyHostToDevice, stream);
		validity = true;
	}
	void destroy() {
		cudaFreeHost(v_cpu[0]);
		cudaFreeHost(v_raw[0]);
		cudaFree(v_gpu_old[0]);
		cudaFree(v_gpu_new[0]);
	}
};

constexpr int NVECS = 9;
#define POS 0
#define VEL 3
#define ACC 6
vec<NVECS> vec_all;
double** pos = &vec_all.v_raw[POS];
double** vel = &vec_all.v_raw[VEL];
double** acc = &vec_all.v_raw[ACC];

constexpr int NSCALS = 2;
#define ENRG 0
#define DEDV 1
vec<NSCALS> scal_all;
double*& energy = scal_all.v_raw[ENRG];
double*& dedv   = scal_all.v_raw[DEDV];

static properties* props;
double potential_energy = 0;
double kinetic_energy = 0;
double temperature = 0;
double total_energy = 0;

void alloc() {
	cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeFourByte);
	cudaStreamCreate(&stream);

	vec_all.init();
	scal_all.init();

	cudaMalloc(&props, ELEMS_NUM * sizeof(properties));
	properties* _props = (properties*)malloc(ELEMS_NUM * sizeof(properties));
	for(int i = 0; i < ELEMS_NUM; i++) _props[i].set_properties(ELEMS_TYPES[i]);
	cudaMemcpy(props, _props, ELEMS_NUM * sizeof(properties), cudaMemcpyHostToDevice);
	free(_props);
}

void dealloc() {
	cudaStreamDestroy(stream);
	vec_all.destroy();
	scal_all.destroy();
	cudaFree(props);
	cudaDeviceReset();
}

void pull_values() {
	vec_all.get_all();
	scal_all.get_all();
}

void push_values() {
	vec_all.set_all();
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
__host__ __device__ float hypotf2(float3 p) {
	return p.x * p.x + p.y * p.y + p.z * p.z;
}

__device__ double3 round(double3 a) {
	return { round(a.x),round(a.y),round(a.z) };
}
__device__ float3 roundf(float3 a) {
	return { roundf(a.x),roundf(a.y),roundf(a.z) };
}

#ifndef __HCC__
__device__ double3& operator+=(double3& a, double3 b) {
	a.x += b.x;
	a.y += b.y;
	a.z += b.z;
	return a;
}
#endif
__device__ double3 operator* (double b, double3 a) {
	return { a.x * b, a.y * b, a.z * b };
}
__device__ double3 operator+ (double3 a, double3 b) {
	return { a.x + b.x, a.y + b.y, a.z + b.z };
}

#ifndef __HCC__
__device__ float3& operator-= (float3& a, float3 b) {
	a.x -= b.x;
	a.y -= b.y;
	a.z -= b.z;
	return a;
}
#endif
__device__ double3& operator+=(double3& a, float3 b) {
	a.x += b.x;
	a.y += b.y;
	a.z += b.z;
	return a;
}
__device__ float3 operator- (float3 a, float3 b) {
	return { a.x - b.x, a.y - b.y, a.z - b.z };
}
__device__ float3 operator* (float b, float3 a) {
	return { a.x * b, a.y * b, a.z * b };
}

__device__ float3 to_f3(double3 a) {
	return {(float)a.x, (float)a.y, (float)a.z};
}

#define GPU_PAIR_INTERACTION_WRAPPER(__COEFFS__, __INIT__, __BODY__, __POST__)  \
    int tid = threadIdx.x,                                                      \
    bid = blockIdx.x,                                                           \
    ind = bid * blockDim.x + tid;                                               \
                                                                                \
    double3 p = 1. / SIZE * vec_all.get(ind, POS),                              \
    v = vec_all.get(ind, VEL);                                                  \
                                                                                \
    float3 p_f = to_f3(p);                                                      \
    properties _P0 = props[get_elem(bid, props[0])];                            \
                                                                                \
    double lj_coeff[ELEMS_NUM];                                                 \
    double em_coeff[ELEMS_NUM];                                                 \
    float ss_ss[ELEMS_NUM];                                                     \
    for(int i = 0; i < ELEMS_NUM; i++) {                                        \
        properties _P = props[i];                                               \
        double epsilon = sqrt(_P.EPSILON * _P0.EPSILON);                        \
        double sigma = (_P.SIGMA + _P0.SIGMA) / 2;                              \
        ss_ss[i] = (SIZE * SIZE) / (sigma * sigma);                             \
        __COEFFS__                                                              \
    }                                                                           \
    extern __shared__ float shm[];                                              \
    float* _posx = shm;                                                         \
    float* _posy = &shm[blockDim.x];                                            \
    float* _posz = &shm[blockDim.x * 2];                                        \
    int props_ind = 0;                                                          \
    for (int i = 0; i < gridDim.x; i++) {                                       \
                                                                                \
        __syncthreads();                                                        \
        float3 _pos = to_f3(1. / SIZE * vec_all.get(i * blockDim.x + tid, POS));\
        _posx[tid] = _pos.x; _posy[tid] = _pos.y; _posz[tid] = _pos.z;          \
                                                                                \
        if(!invalid_elem(i, _P0, props_ind + 1)) props_ind++;                   \
        __INIT__                                                                \
                                                                                \
        __syncthreads();                                                        \
        for (int j = 0; j < blockDim.x; j++) {                                  \
            float3 _p = float3({_posx[j],_posy[j],_posz[j]});                   \
            if (i != bid || j != tid) {                                         \
                __BODY__                                                        \
            }                                                                   \
        }                                                                       \
        __POST__                                                                \
    }                                                                           \
                                                                                \
    p = SIZE * p;

constexpr float ss_rr = (float)((SIZE * SIZE) / (R0 * R0));
constexpr float rr_ss = 1 / ss_rr;
__device__ void get_a(double3& a_lj, double3& a_em, float3 p, float3 _p, float ss_ss) {
	float3 d = p - _p;
	d -= roundf(d);

	float d2 = hypotf2(d);

#ifdef ENABLE_LJ
	float r2 = d2 * ss_ss,
		r_2 = 1.f / r2,
		r_4 = r_2 * r_2,
		r_6 = r_4 * r_2,
		r_8 = r_4 * r_4,
		_2r_14__r_8 = (r_6 - .5f) * r_8;
	a_lj += (_2r_14__r_8 * d);
#endif

#ifdef ENABLE_EM
	if (d2 < rr_ss)
		d2 = rr_ss;
	float d_1 = rsqrtf(d2);
	float em_coeff = d_1 * d_1 * d_1;
	a_em += (em_coeff * d);
#endif
}

constexpr float
	c1 = (float)_sqrt(ss_rr) * 1.5f,
	c2 = -(float)_sqrt(ss_rr) * ss_rr * 0.5f;

__device__ void get_e(double& e_lj, double& e_em, const float3 p, const float3 _p, float ss_ss, double& dedv_lj, double& dedv_em) {
	float3 d = p - _p;
	d -= roundf(d);

	float d2 = hypotf2(d);

#ifdef ENABLE_LJ
	float r2 = d2 * ss_ss,
		r_2 = 1.f / r2,
		r_4 = r_2 * r_2,
		r_6 = r_4 * r_2;
	e_lj += (r_6 - 1.f) * r_6;
	static_assert(0, "dedv_lj not implemented");
#endif

#ifdef ENABLE_EM
	float e_em_const = 0.f;
	float e_em_poly = 0.f;
	float e_em_coeff = -0.5f;

	if (d2 < rr_ss) {
		e_em_const = c1;
		e_em_poly = c2 * d2;
		e_em_coeff = 1;
	}
	else
		e_em_poly = rsqrtf(d2);
	e_em += e_em_const + e_em_poly;
	dedv_em += e_em_poly * e_em_coeff;

#endif
}

__global__
void euler_gpu(const vec<NVECS> vec_all, properties* props) {
	double3 a_lj = d3_0;
	double3 a_em = d3_0;

	GPU_PAIR_INTERACTION_WRAPPER(
		lj_coeff[i] = 48. * epsilon * SIZE / sigma / sigma / _P0.M;
		em_coeff[i] = 1. / (4. * PI * EPSILON0) / SIZE / SIZE * _P0.Q * _P.Q / _P0.M;
	,
		double3 da_lj = d3_0;
		double3 da_em = d3_0;
	,
		get_a(da_lj, da_em, p_f, _p, ss_ss[props_ind]);
	,
		a_lj += lj_coeff[props_ind] * da_lj;
		a_em += em_coeff[props_ind] * da_em;
	)

	double3 a = a_lj + a_em;

	v += TIME_STEP * a;
	p += TIME_STEP * v;

	vec_all.set(ind, p, POS);
	vec_all.set(ind, v, VEL);
	vec_all.set(ind, a, ACC);

}
__global__
void energy_gpu (const vec<NVECS> vec_all, const vec<NSCALS> scal_all, properties* props) {
	double e_lj = 0;
	double e_em = 0;

	GPU_PAIR_INTERACTION_WRAPPER(
		lj_coeff[i] = 2. * epsilon;
		em_coeff[i] = 1. / (8. * PI * EPSILON0) * _P0.Q * _P.Q / SIZE;
	,
		double de_lj = 0;
		double de_em = 0;
		double ddedv_lj = 0;
		double ddedv_em = 0;
	,
		get_e(de_lj, de_em, p_f, _p, ss_ss[props_ind], ddedv_lj, ddedv_em);
	,
		e_lj += lj_coeff[props_ind] * de_lj;
		e_em += em_coeff[props_ind] * de_em;
	);

	scal_all.set_single(ind, e_em + e_lj, ENRG);
}

void energy_calc() {
	potential_energy = 0;
	kinetic_energy = 0;
	for (int i = 0; i < AMOUNT; i++) {
		potential_energy += energy[i] / AMOUNT;
		kinetic_energy += get_properties(i).M * hypot2(extract(vel, i)) / 2 / AMOUNT;
	}
	total_energy = potential_energy + kinetic_energy;
}

void euler_steps(int steps) {
	for(int i = 0; i < steps; i++) {
	#ifndef __INTELLISENSE__
		euler_gpu <<< GRID_SIZE, BLOCK_SIZE, sizeof(float) * BLOCK_SIZE * 3, stream >>> (vec_all, props);
	#endif
		vec_all.gpu_copy();
	}
	vec_all.invalidate();

#ifndef __INTELLISENSE__
	energy_gpu <<< GRID_SIZE, BLOCK_SIZE, sizeof(float) * BLOCK_SIZE * 3, stream >>> (vec_all, scal_all, props);
#endif
	scal_all.invalidate();

	energy_calc();
}

void force_energy_calc() {
#ifndef __INTELLISENSE__
	energy_gpu <<< GRID_SIZE, BLOCK_SIZE, sizeof(float) * BLOCK_SIZE * 3 >>> (vec_all, scal_all, props);
#endif
	scal_all.invalidate();
	scal_all.get_all();
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
	printf("regsPerBlock: %d\n", chars.regsPerBlock);
	printf("warpSize: %d\n\n", chars.warpSize);

#ifndef __HIPCC__
	printf("singleToDoublePrecisionPerfRatio: %d\n", chars.singleToDoublePrecisionPerfRatio);
	printf("kernelExecTimeoutEnabled: %d\n", chars.kernelExecTimeoutEnabled);
	printf("regsPerMultiprocessor: %d\n", chars.regsPerMultiprocessor);
	printf("sharedMemPerMultiprocessor: %zu\n", chars.sharedMemPerMultiprocessor);
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
	cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocks, (const void*)euler_gpu, BLOCK_SIZE, sizeof(float) * BLOCK_SIZE * 3);
	printf("BlockSize = %d; BlocksPerMP = %d; Occupancy = %f\n", BLOCK_SIZE, numBlocks, (double) (numBlocks * BLOCK_SIZE) / (chars.maxThreadsPerMultiProcessor));
	printf("\n");
#endif
}
