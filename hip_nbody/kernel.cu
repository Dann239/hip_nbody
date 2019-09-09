#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "butchers.cuh"
#include "kernel.h"

#include <iostream>
using namespace std;

static double* p[3];
static double* v[3];

void gpu_alloc() {
	for (int i = 0; i < 3; i++) {
		cudaMalloc(&p[i], AMOUNT * sizeof(double));
		cudaMalloc(&v[i], AMOUNT * sizeof(double));
	}
}
void gpu_dealloc() {
	for (int i = 0; i < 3; i++) {
		cudaFree(p[i]);
		cudaFree(v[i]);
	}
	cudaDeviceReset();
}

void get_pos(double* _p[3]) {
	for(int i = 0; i < 3; i++)
		cudaMemcpy(_p[i], p[i], AMOUNT * sizeof(double), cudaMemcpyDeviceToHost);
}
void get_vel(double* _v[3]) {
	for (int i = 0; i < 3; i++)
		cudaMemcpy(_v[i], v[i], AMOUNT * sizeof(double), cudaMemcpyDeviceToHost);
}

void set_pos(double* _p[3]) {
	for (int i = 0; i < 3; i++)
		cudaMemcpy(p[i], _p[i], AMOUNT * sizeof(double), cudaMemcpyHostToDevice);
}
void set_vel(double* _v[3]) {
	for (int i = 0; i < 3; i++)
		cudaMemcpy(v[i], _v[i], AMOUNT * sizeof(double), cudaMemcpyHostToDevice);
}

void print_err() {
	cudaDeviceSynchronize();
	cout << cudaGetErrorString(cudaGetLastError()) << endl;
}
