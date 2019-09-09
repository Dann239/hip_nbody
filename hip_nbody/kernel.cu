#include "cuda_runtime.h"

#ifndef __HIPCC__
#include "device_launch_parameters.h"
#endif

#include "butchers.cuh"
#include "kernel.h"

#include <iostream>
using namespace std;

static double* pos[3];
static double* vel[3];

void gpu_alloc() {
	for (int i = 0; i < 3; i++) {
		cudaMalloc(&pos[i], AMOUNT * sizeof(double));
		cudaMalloc(&vel[i], AMOUNT * sizeof(double));
	}
}
void gpu_dealloc() {
	for (int i = 0; i < 3; i++) {
		cudaFree(pos[i]);
		cudaFree(vel[i]);
	}
	cudaDeviceReset();
}

void get_pos(double* _pos[3]) {
	for(int i = 0; i < 3; i++)
		cudaMemcpy(_pos[i], pos[i], AMOUNT * sizeof(double), cudaMemcpyDeviceToHost);
}
void get_vel(double* _vel[3]) {
	for (int i = 0; i < 3; i++)
		cudaMemcpy(_vel[i], vel[i], AMOUNT * sizeof(double), cudaMemcpyDeviceToHost);
}

void set_pos(double* _pos[3]) {
	for (int i = 0; i < 3; i++)
		cudaMemcpy(pos[i], _pos[i], AMOUNT * sizeof(double), cudaMemcpyHostToDevice);
}
void set_vel(double* _vel[3]) {
	for (int i = 0; i < 3; i++)
		cudaMemcpy(vel[i], _vel[i], AMOUNT * sizeof(double), cudaMemcpyHostToDevice);
}

void print_err() {
	cudaDeviceSynchronize();
	cout << cudaGetErrorString(cudaGetLastError()) << endl;
}
