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
	__device__ double get_single(int i, int offset) const {
		return v_gpu_old[offset][i];
	}
	__device__ void set_single(int i, double p, int offset) const {
		v_gpu_new[offset][i] = p;
	}
	void gpu_copy(int begin = 0, int len = size) {
		cudaMemcpyAsync(v_gpu_old[begin], v_gpu_new[begin], MEM_LEN * len, cudaMemcpyDeviceToDevice, stream);
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
		cudaMemcpyAsync(v_gpu_new[0], v_raw[0], MEM_LEN * size, cudaMemcpyHostToDevice, stream);
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
#define THETA 6
#define ENRG 7
#define VIRI 8
vec<NVECS> vec_all;
double** pos = &vec_all.v_raw[POS];
double** vel = &vec_all.v_raw[VEL];
double*& theta = vec_all.v_raw[THETA];
double*& enrg = vec_all.v_raw[ENRG];
double*& viri = vec_all.v_raw[VIRI];

static properties* props;
void alloc() {
	cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);
	cudaStreamCreate(&stream);

	vec_all.init();
	
	cudaMalloc(&props, ELEMS_NUM * sizeof(properties));
	properties* _props = (properties*)malloc(ELEMS_NUM * sizeof(properties));
	for(int i = 0; i < ELEMS_NUM; i++) _props[i].set_properties(ELEMS_TYPES[i]);
	cudaMemcpy(props, _props, ELEMS_NUM * sizeof(properties), cudaMemcpyHostToDevice);
	free(_props);
}
void dealloc() {
	cudaStreamDestroy(stream);
	vec_all.destroy();
	cudaFree(props);
	cudaDeviceReset();
}

void pull_values() {
	vec_all.get_all();
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
ELEMS get_elem_type(int num) {
	return ELEMS_TYPES[get_elem(num / BLOCK_SIZE, properties(ERROR))];
}
properties get_properties(int num) {
	return properties(get_elem_type(num));
}
__host__ __device__ double hypot2(double3 p) {
	return p.x * p.x + p.y * p.y + p.z * p.z;
}
__device__ double3 round(double3 a) {
	return { round(a.x),round(a.y),round(a.z) };
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
__device__ double3 operator- (double3 a, double3 b) {
	return { a.x - b.x, a.y - b.y, a.z - b.z };
}
#ifndef __HCC__
__device__ double3& operator-= (double3& a, double3 b) {
	a.x -= b.x;
	a.y -= b.y;
	a.z -= b.z;
	return a;
}
#endif
__device__ double operator* (double3 a, double3 b) {
	return a.x * b.x + a.y * b.y + a.z * b.z;
}

#define GPU_PAIR_INTERACTION_WRAPPER(__COEFFS__, __INIT__, __BODY__, __POST__)        \
    int tid = threadIdx.x,                                                            \
    bid = blockIdx.x,                                                                 \
    ind = bid * blockDim.x + tid;                                                     \
                                                                                      \
    double3 p = 1. / SIZE * vec_all.get(ind, POS),                                    \
    v = vec_all.get(ind, VEL);                                                        \
    double th = vec_all.get_single(ind, THETA);                                       \
    double logth = log(th);                                                           \
                                                                                      \
    properties _P0 = props[get_elem(bid, props[0])];                                  \
                                                                                      \
    double lj_coeff[ELEMS_NUM];                                                       \
    double ss_ss[ELEMS_NUM];                                                          \
                                                                                      \
    for(int i = 0; i < ELEMS_NUM; i++) {                                              \
                                                                                      \
        properties _P = props[i];                                                     \
        double epsilon = sqrt(_P.EPSILON * _P0.EPSILON);                              \
        double sigma = (_P.SIGMA + _P0.SIGMA) / 2;                                    \
        ss_ss[i] = (SIZE * SIZE) / (sigma * sigma);                                   \
                                                                                      \
        __COEFFS__                                                                    \
    }                                                                                 \
                                                                                      \
    extern __shared__ double shm[];                                                   \
    double* _posx = shm;                                                              \
    double* _posy = &shm[blockDim.x];                                                 \
    double* _posz = &shm[blockDim.x * 2];                                             \
    int props_ind = 0;                                                                \
    for (int i = 0; i < gridDim.x; i++) {                                             \
                                                                                      \
        __syncthreads();                                                              \
        double3 _pos = 1. / SIZE * vec_all.get(i * blockDim.x + tid, POS);            \
        _posx[tid] = _pos.x; _posy[tid] = _pos.y; _posz[tid] = _pos.z;                \
                                                                                      \
        if(!invalid_elem(i, _P0, props_ind + 1)) props_ind++;                         \
        __INIT__                                                                      \
                                                                                      \
        __syncthreads();                                                              \
        for (int j = 0; j < blockDim.x; j++) {                                        \
            double3 _p = double3({_posx[j],_posy[j],_posz[j]});                       \
            if (i != bid || j != tid) {                                               \
                __BODY__                                                              \
            }                                                                         \
        }                                                                             \
        __POST__                                                                      \
    }                                                                                 \
                                                                                      \
    p = SIZE * p;                                                                     \


__device__ void get_a(double3& a_lj, double3& a_eam, double3 d, const double ss_ss, const double log_th, const double beta, const double c1) {
#ifdef ENABLE_PB
	d -= round(d);
#endif

	double d2 = hypot2(d);
	double r2 = d2 * ss_ss;

#ifdef ENABLE_LJ
	double r_2 = 1 / r2,
		r_4 = r_2 * r_2,
		r_6 = r_4 * r_2,
		r_8 = r_4 * r_4,
		_2r_14__r_8 = (r_6 - 1) * r_8;
	a_lj += (_2r_14__r_8 * d);


#endif

#ifdef ENABLE_EAM
	double _r = rsqrt(r2);
	double r = 1 / _r;
	a_eam += (log_th + beta * r) * exp(-beta * r) * _r * d; //eam
	//a_eam += (beta * (r - 1)) * exp(-beta * r) * _r * d; //eam pairwise
	//a_eam += (log_th + beta) * exp(-beta * r) * _r * d; //eam embedding
#endif
}

__global__
void euler_gpu(const vec<NVECS> vec_all, properties* props, double beta, double A) {
	double3 a_lj = d3_0;
	double3 a_eam = d3_0;

	extern __shared__ double _shm[];
	double* _thetas = &_shm[blockDim.x * 3];
	double c1, c2;
	
	GPU_PAIR_INTERACTION_WRAPPER(
		lj_coeff[i] = 0.25 * 48. * SIZE * epsilon / sigma / sigma / _P0.M;
		c1 = -log(Z0);
		c2 = A * beta * exp(beta) * SIZE;
	,
		double3 da_lj = d3_0;
		double3 da_eam = d3_0;
		_thetas[tid] = (log(vec_all.get_single(i * blockDim.x + tid, THETA)) + logth) * 0.5 + c1;
	,
		get_a(da_lj, da_eam, p - _p, ss_ss[props_ind], _thetas[j], beta, c1);
	,
		a_lj += lj_coeff[props_ind] * da_lj;
		a_eam += c2 * da_eam;
	);

	double3 a = a_lj + a_eam;

	v += TIME_STEP * a;
	p += TIME_STEP * v;

	vec_all.set(ind, p, POS);
	vec_all.set(ind, v, VEL);
}

__device__ void get_e(double& e_lj, double& e_eam, double3 d, const double ss_ss, const double beta, const double c1) {
#ifdef ENABLE_PB
	d -= round(d);
#endif

	double d2 = hypot2(d);
	double r2 = d2 * ss_ss;

#ifdef ENABLE_LJ
	double r_2 = 1 / r2,
		r_4 = r_2 * r_2,
		r_6 = r_4 * r_2;
	e_lj += (r_6 - 2) * r_6;
#endif
#ifdef ENABLE_EAM
	double r = sqrt(r2);
	e_eam += exp(-beta * r) * (r + c1);
#endif

}

__global__
void energy_gpu (const vec<NVECS> vec_all, properties* props, double beta, double A) {
	double e_lj = 0;
	double e_eam = 0;
	
	double c1 = -1 + 1 / beta;
	double eam_coeff;

	GPU_PAIR_INTERACTION_WRAPPER(
		lj_coeff[i] = 0.25 * 2. * epsilon;
		eam_coeff = A * beta * exp(beta) / 2;
	,
		double de_lj = 0;
		double de_eam = 0;
	,
		get_e(de_lj, de_eam, p - _p, ss_ss[props_ind], beta, c1);
	,
		e_lj += lj_coeff[props_ind] * de_lj;
		e_eam += eam_coeff * de_eam; //eam pairwise
	);
#ifdef ENABLE_EAM
	e_eam += eam_coeff / beta * th * (beta - log(Z0) - 1 + log(th)); //eam embedding
#endif

	vec_all.set_single(ind, e_lj + e_eam, ENRG);

}

__device__ void get_viri(double& v_lj, double& v_eam, double3 d, const double ss_ss, double beta, double log_th) {
#ifdef ENABLE_PB
	d -= round(d);
#endif

	double d2 = hypot2(d);

#ifdef ENABLE_LJ
	double r2 = d2 * ss_ss,
		r_2 = 1.f / r2,
		r_4 = r_2 * r_2,
		r_6 = r_4 * r_2,
		r_8 = r_4 * r_4,
		_2r_14__r_8 = (r_6 - .5f) * r_8;
	double3 da_lj = (_2r_14__r_8 * d);
	v_lj += (da_lj * d);
#endif

#ifdef ENABLE_EAM
	double _r = rsqrt(r2);
	double r = 1 / _r;
	v_eam += (log_th + beta * r) * exp(-beta * r) * _r * d * d; //eam
	//a_eam += (beta * (r - 1)) * exp(-beta * r) * _r * d; //eam pairwise
	//a_eam += (log_th + beta) * exp(-beta * r) * _r * d; //eam embedding
#endif

}
__global__ void viri_gpu(const vec<NVECS> vec_all, properties* props, double A, double beta) {
	double v_lj = 0;
	double v_eam = 0;

	extern __shared__ double _shm[];
	double* _thetas = &_shm[blockDim.x * 3];

	double eam_coeff, c1;

	GPU_PAIR_INTERACTION_WRAPPER(
		lj_coeff[i] = 0.25 * 48. * SIZE * SIZE * epsilon / sigma / sigma / (3. * V) / 2.;
		eam_coeff = A * beta * exp(beta) * SIZE / (3. * V) / 2.;
		c1 = -log(Z0);
	,
		double dv_lj = 0;
		double dv_eam = 0;
		_thetas[tid] = (log(vec_all.get_single(i * blockDim.x + tid, THETA)) + logth) * 0.5 + c1;
	,
		get_viri(dv_lj, dv_eam, p - _p, ss_ss[props_ind], beta, _thetas[tid]);
	,
		v_lj += lj_coeff[props_ind] * dv_lj;
		v_eam += eam_coeff * dv_eam;
	);

	vec_all.set_single(ind, v_lj, VIRI);
}

__device__ void get_th(double& th, double3 d, const double coeff) {
#ifdef ENABLE_PB
	d -= round(d);
#endif

	th += exp(sqrt(hypot2(d)) * coeff);
}
__global__ void theta_gpu(const vec<NVECS> vec_all, properties* props, double beta) {
	double theta = 0;
	double coeff = -SIZE * beta;

	GPU_PAIR_INTERACTION_WRAPPER(
		;
	,
		;
	,
		get_th(theta, p - _p, coeff);
	,
		;
	);

	vec_all.set_single(ind, theta, THETA);
}

double total_time = 0;
void euler_steps(int steps) {
	for(int i = 0; i < steps; i++) {
#ifndef __INTELLISENSE__
		theta_gpu << < GRID_SIZE, BLOCK_SIZE, sizeof(double)* BLOCK_SIZE * 3, stream >> > (vec_all, props, BETA);
#endif
		vec_all.gpu_copy(6, 1);
#ifndef __INTELLISENSE__
		euler_gpu <<< GRID_SIZE, BLOCK_SIZE, sizeof(double) * BLOCK_SIZE * 4, stream >>> (vec_all, props, BETA, A);
#endif
		vec_all.gpu_copy(0, 6);
	}
	total_time += TIME_STEP * steps;
	force_energy_calc();
}

void force_energy_calc() {
#ifndef __INTELLISENSE__
	theta_gpu << < GRID_SIZE, BLOCK_SIZE, sizeof(double)* BLOCK_SIZE * 3, stream >> > (vec_all, props, BETA);
	vec_all.gpu_copy(6, 1);
	energy_gpu <<< GRID_SIZE, BLOCK_SIZE, sizeof(double) * BLOCK_SIZE * 3, stream >>> (vec_all, props, BETA, A);
	viri_gpu <<< GRID_SIZE, BLOCK_SIZE, sizeof(double) * BLOCK_SIZE * 4, stream >>> (vec_all, props, BETA, A);
#endif
	vec_all.invalidate();
}

bool selectDevice(int deviceIndex) {
	int devicesCount;
	cudaGetDeviceCount(&devicesCount);

	cudaDeviceProp deviceProperties;
	cudaGetDeviceProperties(&deviceProperties, deviceIndex);

	if (deviceProperties.major >= 6 && deviceProperties.minor >= 0) {
            cudaSetDevice(deviceIndex);
            return true;
        }

	return false;
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
	cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocks, (const void*)euler_gpu, BLOCK_SIZE, sizeof(double) * BLOCK_SIZE * 3);
	printf("BlockSize = %d; BlocksPerMP = %d; Occupancy = %f\n", BLOCK_SIZE, numBlocks, (double) (numBlocks * BLOCK_SIZE) / (chars.maxThreadsPerMultiProcessor));
	printf("\n");
#endif
}
