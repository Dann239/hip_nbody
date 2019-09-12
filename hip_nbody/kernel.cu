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

static double* pos[3];
static double* vel[3];
static double* acc[3];
static double* energy;
static properties* props;

struct vec {
	double* v[3];
	__device__ void set(int i, double3 p) {
		v[X][i] = p.x;
		v[Y][i] = p.y;
		v[Z][i] = p.z;
	}
	__device__ double3 get(int i) {
		return double3({ v[X][i],v[Y][i],v[Z][i] });
	}
	vec(double* p[3]) {
		for (int i = 0; i < 3; i++)
			v[i] = p[i];
	}
};

properties::properties(int block) {
	for (int i = 0; i < ELEMS_NUM - 1; i++)
		if (1. * block / GRID_SIZE < ELEMS_DIVISIONS[i]) {
			*this = properties((ELEMS)ELEMS_TYPES[i]);
			return;
		}
	*this = properties((ELEMS)ELEMS_TYPES[ELEMS_NUM - 1]);
}

void gpu_alloc() {
	for (int i = 0; i < 3; i++) {
		cudaMalloc(&pos[i], AMOUNT * sizeof(double));
		cudaMalloc(&vel[i], AMOUNT * sizeof(double));
		cudaMalloc(&acc[i], AMOUNT * sizeof(double));
		cudaMemset(acc[i], 0, AMOUNT * sizeof(double));
	}
	cudaMalloc(&energy, AMOUNT * sizeof(double));

	cudaMalloc(&props, GRID_SIZE * sizeof(properties));
	static properties* _props = new properties[GRID_SIZE];
	for (int i = 0; i < GRID_SIZE; i++)
		_props[i] = properties(i);
	cudaMemcpy(props, _props, GRID_SIZE * sizeof(properties), cudaMemcpyHostToDevice);
}
void gpu_dealloc() {
	for (int i = 0; i < 3; i++) {
		cudaFree(pos[i]);
		cudaFree(vel[i]);
		cudaFree(acc[i]);
	}
	cudaFree(energy);
	cudaDeviceReset();
}

bool pos_valid = false, vel_valid = false, energy_valid = false;

void get_pos() {
	if (pos_valid)
		return;
	for(int i = 0; i < 3; i++)
		cudaMemcpy(_pos[i], pos[i], AMOUNT * sizeof(double), cudaMemcpyDeviceToHost);
	pos_valid = true;
}
void get_vel() {
	if (vel_valid)
		return;
	for (int i = 0; i < 3; i++)
		cudaMemcpy(_vel[i], vel[i], AMOUNT * sizeof(double), cudaMemcpyDeviceToHost);
	vel_valid = true;
}

void set_pos() {
	for (int i = 0; i < 3; i++)
		cudaMemcpy(pos[i], _pos[i], AMOUNT * sizeof(double), cudaMemcpyHostToDevice);
	pos_valid = true;
}
void set_vel() {
	for (int i = 0; i < 3; i++)
		cudaMemcpy(vel[i], _vel[i], AMOUNT * sizeof(double), cudaMemcpyHostToDevice);
	vel_valid = true;
}

void print_err() {
	cudaDeviceSynchronize();
	cout << cudaGetErrorString(cudaGetLastError()) << endl;
}

__device__ double hypot2(double3 p) {
	return p.x * p.x + p.y * p.y + p.z * p.z;
}
__device__ double3 round(double3 a) {
	return { round(a.x),round(a.y),round(a.z) };
}

#ifndef __HIPCC__
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

__device__ bool operator== (double3 a, double3 b) {
	return a.x == b.x && a.y == b.y && a.z == b.z;
}

//constexpr double ss_ss = (SIZE * SIZE) / (SIGMA * SIGMA);

__device__ properties combine(properties p, properties _p) {
	p.EPSILON = sqrt(p.EPSILON * _p.EPSILON);
	p.SIGMA = (p.SIGMA + _p.SIGMA) / 2;
	p.Q = sqrt(p.Q * _p.Q);
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
	ind = bid * BLOCK_SIZE + tid;															\
																							\
	double3 p = 1. / SIZE * pos.get(ind),													\
	v = vel.get(ind);																		\
																							\
	properties P = props[0];																\
	__shared__ double3 _pos[BLOCK_SIZE];													\
	for (int i = 0; i < GRID_SIZE; i++) {													\
		__syncthreads();																	\
		_pos[tid] = 1. / SIZE * pos.get(i * BLOCK_SIZE + tid);								\
		__syncthreads();																	\
																							\
		properties _P = combine(P, props[0]);												\
		double ss_ss = (SIZE * SIZE) / (_P.SIGMA * _P.SIGMA);								\
																							\
		__INIT__																			\
		for (int j = 0; j < BLOCK_SIZE; j++) {												\
			double3 _p = _pos[j];															\
			if (i != bid || j != tid) {														\
				__BODY__																	\
			}																				\
		}																					\
		__POST__																			\
	}

__global__ void euler_gpu(vec pos, vec vel, vec acc, properties* props) {
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
	
	p *= SIZE;
	
	double3 _a = acc.get(ind);
	double3 a = a_lj + a_em;
	acc.set(ind, a);

	vel.set(ind, v + TIME_STEP * a);
	pos.set(ind, p + TIME_STEP * (v + TIME_STEP * a));
}

__global__ void energy_gpu(vec pos, vec vel, double* energy, properties* props) {
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

	//e_lj *= 2. * EPSILON;
	//e_em *= 1. / (8. * PI * EPSILON0) * Q * Q / SIZE;
	
	double e_k = P.M * hypot2(v) / 2.;
	energy[ind] = e_k + e_em + e_lj;
}


double get_energy() {
	static double total_energy = 0;
	if (energy_valid)
		return total_energy;
	energy_valid = true;

#ifndef __INTELLISENSE__
	energy_gpu <<< GRID_SIZE, BLOCK_SIZE >>> (pos, vel, energy, props);
#endif 

	static double _energy[AMOUNT];
	total_energy = 0;
	cudaMemcpy(_energy, energy, AMOUNT * sizeof(double), cudaMemcpyDeviceToHost);

	for (int i = 0; i < AMOUNT; i++) {
		total_energy += _energy[i];
	}

	return total_energy;
}
void euler_step() {

#ifndef __INTELLISENSE__
	euler_gpu << < GRID_SIZE, BLOCK_SIZE >> > (pos, vel, acc, props);
#endif

	pos_valid = vel_valid = energy_valid = false;
}
