#include "kernel.h"

#include <vector>
#include <iostream>
#include <fstream>
#include <time.h>
using namespace std;

static double* pos[3];
static double* vel[3];

void cpu_alloc() {
	for (int i = 0; i < 3; i++) {
		pos[i] = new double[AMOUNT];
		vel[i] = new double[AMOUNT];
	}
}

void randomize() {
	constexpr int grid_size = (int)(_cbrt(AMOUNT) + 1);
	vector<vector<vector<bool> > > grid(grid_size, vector<vector<bool> >(grid_size, vector<bool>(grid_size, false)));
	for (int i = 0; i < grid_size; i++)
		for (int j = 0; j < grid_size; j++)
			for (int k = 0; k < grid_size; k++)
				grid[i][j][k] = false;

	for (int i = 0; i < AMOUNT; i++) {
		int p[3];
		for (int j = 0; j < 3; j++) {
			p[j] = rand() % grid_size;
			pos[j][i] = p[j] * SIZE / grid_size;
			vel[j][i] = ((double)rand() / RAND_MAX - .5) * 2 * V;
		}

		if (grid[p[0]][p[1]][p[2]]) {
			i--;
			continue;
		}

		grid[p[0]][p[1]][p[2]] = true;
	}

	set_pos(pos);
	set_vel(vel);
}

void dump() {
	get_pos(pos);
	get_vel(vel);
	ofstream out("data/dump.dat", ios::binary);
	for (int i = 0; i < 3; i++) {
		out.write((char*)pos[i], AMOUNT * sizeof(double));
		out.write((char*)vel[i], AMOUNT * sizeof(double));
	}
}
void load() {
	ifstream in("data/dump.dat", ios::binary);
	if (!in.fail()) {
		for (int i = 0; i < 3; i++) {
			in.read((char*)pos[i], AMOUNT * sizeof(double));
			in.read((char*)vel[i], AMOUNT * sizeof(double));
		}
		set_pos(pos);
		set_vel(vel);
	}
}

int main() {
	cpu_alloc();
	gpu_alloc();
	randomize();
	const int steps = 10;
	long long t = clock();
	for (int i = 0; i < steps; i++) {
		euler_step();
	cout << get_energy() << endl;
	}

	cout << (27. * AMOUNT * AMOUNT * steps / ((long long)(clock() - t) / CLOCKS_PER_SEC) * 1e-12) << "TFlops" << endl;
	
	gpu_dealloc();
	return 0;
}