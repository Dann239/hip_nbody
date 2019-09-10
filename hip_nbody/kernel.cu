#include "cuda_runtime.h"

#ifndef __HIPCC__
#include "device_launch_parameters.h"
#include "cuda_runtime_api.h"
#endif

#ifdef __INTELLISENSE__
void __syncthreads() {}
#endif

#include "kernel.h"

#include <iostream>
using namespace std;

#define POS(i) double3({posx[i], posy[i], posz[i]})
#define VEL(i) double3({velx[i], vely[i], velz[i]})

static double* pos[3];
static double* vel[3];
static double* energy;

void gpu_alloc() {
	for (int i = 0; i < 3; i++) {
		cudaMalloc(&pos[i], AMOUNT * sizeof(double));
		cudaMalloc(&vel[i], AMOUNT * sizeof(double));
	}
	cudaMalloc(&energy, AMOUNT * sizeof(double));
}
void gpu_dealloc() {
	for (int i = 0; i < 3; i++) {
		cudaFree(pos[i]);
		cudaFree(vel[i]);
	}
	cudaFree(energy);
	cudaDeviceReset();
}

bool pos_valid = false, vel_valid = false, energy_valid = false;

void get_pos(double* _pos[3]) {
	if (pos_valid)
		return;
	for(int i = 0; i < 3; i++)
		cudaMemcpy(_pos[i], pos[i], AMOUNT * sizeof(double), cudaMemcpyDeviceToHost);
	pos_valid = true;
}
void get_vel(double* _vel[3]) {
	if (vel_valid)
		return;
	for (int i = 0; i < 3; i++)
		cudaMemcpy(_vel[i], vel[i], AMOUNT * sizeof(double), cudaMemcpyDeviceToHost);
	vel_valid = true;
}

void set_pos(double* _pos[3]) {
	for (int i = 0; i < 3; i++)
		cudaMemcpy(pos[i], _pos[i], AMOUNT * sizeof(double), cudaMemcpyHostToDevice);
	pos_valid = true;
}
void set_vel(double* _vel[3]) {
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


#define GPU_PAIR_INTERACTION_WRAPPER( __CODE__ ) 				\
																\
	int tid = threadIdx.x,										\
	bid = blockIdx.x,											\
	ind = bid * BLOCK_SIZE + tid;								\
																\
	double3 p = 1. / SIZE * POS(ind),							\
	v = VEL(ind);												\
																\
	__shared__ double3 _pos[BLOCK_SIZE];						\
	for (int i = 0; i < GRID_SIZE; i++) {						\
																\
		__syncthreads();										\
		_pos[tid] = 1. / SIZE * POS(i * BLOCK_SIZE + tid);		\
		__syncthreads();										\
																\
		for (int j = 0; j < BLOCK_SIZE; j++) {					\
			double3 _p = _pos[j];								\
			if (i != bid || j != tid) {							\
				__CODE__										\
			}													\
		}														\
	}															\

constexpr double ss_ss = (SIZE * SIZE) / (SIGMA * SIGMA);

__device__ double3 lj_a(double3 a, double3 p, double3 _p) {
	double3 d = p - _p;
	d -= round(d);

	double r2 = hypot2(d) * ss_ss,
		r_2 = 1. / r2,
		r_4 = r_2 * r_2,
		r_6 = r_4 * r_2,
		r_8 = r_4 * r_4,
		_2r_14__r_8 = (r_6 - .5) * r_8;

	return a + (_2r_14__r_8 * d);
}
__device__ double lj_e(double e, double3 p, double3 _p) {
	double3 d = p - _p;
	d -= round(d);

	double r2 = hypot2(d) * ss_ss,
		r_2 = 1. / r2,
		r_4 = r_2 * r_2,
		r_6 = r_4 * r_2;

	return e + (r_6 - 1) * r_6;
}

__global__ void euler_gpu(double* posx, double* posy, double* posz, double* velx, double* vely, double* velz) {
	double3 a = { 0., 0., 0. };

	GPU_PAIR_INTERACTION_WRAPPER(a = lj_a(a, p, _p);)

	v += ((4. * EPSILON / SIGMA / SIGMA / M) * (12. * SIZE) * TIME_STEP) * a;
	velx[ind] = v.x; vely[ind] = v.y, velz[ind] = v.z;
	v *= TIME_STEP;
	posx[ind] += v.x; posy[ind] += v.y, posz[ind] += v.z;
}
__global__ void energy_gpu(double* posx, double* posy, double* posz, double* velx, double* vely, double* velz, double* energy) {
	double e = 0;
	
	GPU_PAIR_INTERACTION_WRAPPER(e = lj_e(e, p, _p);)

	energy[ind] = e / 2. + M * hypot2(v) / 2.;
}


double get_energy() {
	static double total_energy = 0;
	if (energy_valid)
		return total_energy;
	energy_valid = true;

#ifndef __INTELLISENSE__
	energy_gpu <<< GRID_SIZE, BLOCK_SIZE >>> (pos[0], pos[1], pos[2], vel[0], vel[1], vel[2], energy);
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
	euler_gpu <<< GRID_SIZE, BLOCK_SIZE >>> (pos[0], pos[1], pos[2], vel[0], vel[1], vel[2]);
#endif

	pos_valid = vel_valid = energy_valid = false;
}
