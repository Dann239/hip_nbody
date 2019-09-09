#include <iostream>
#include "kernel.h"
using namespace std;

static double* p[3];
static double* v[3];

void cpu_alloc() {
	for (int i = 0; i < 3; i++) {
		p[i] = new double[AMOUNT];
		v[i] = new double[AMOUNT];
	}
}

int main() {
	cpu_alloc();

	gpu_alloc();
	print_err();

	set_pos(p);
	print_err();
	
	get_pos(p);
	print_err();
	
	gpu_dealloc();
	print_err();
	
	return 0;
}