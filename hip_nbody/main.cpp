#include "kernel.h"
#include "window.h"
#include "properties.h"

#include <vector>
#include <iostream>
#include <fstream>
#include <time.h>
using namespace std;

constexpr double OUTPUT_COEFF = SCREEN_SIZE / SIZE;

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
	bool grid[grid_size][grid_size][grid_size];
	for (int i = 0; i < grid_size; i++)
		for (int j = 0; j < grid_size; j++)
			for (int k = 0; k < grid_size; k++)
				grid[i][j][k] = false;

	for (int i = 0; i < AMOUNT; i++) {
		int grid_pos[3];
		for (int j = 0; j < 3; j++) {
			grid_pos[j] = rand() % grid_size;
			pos[j][i] = grid_pos[j] * SIZE / grid_size;
			vel[j][i] = ((double)rand() / RAND_MAX - .5) * 2 * V;
		}

		if (grid[grid_pos[0]][grid_pos[1]][grid_pos[2]]) {
			i--;
			continue;
		}

		grid[grid_pos[0]][grid_pos[1]][grid_pos[2]] = true;
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
	window_init();
	
	while(window_is_open()) {
		for(int i = 0; i < SKIPS; i++)
			euler_step();
		for (int i = 0; i < AMOUNT; i++) {
			get_pos(pos);
			window_draw_point(pos[0][i] * OUTPUT_COEFF, pos[1][i] * OUTPUT_COEFF);
		}
		window_show();
		cout << get_energy() << endl;
	}
	window_delete();
	
	
	gpu_dealloc();
	return 0;
}