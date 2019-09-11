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

double* _pos[3];
double* _vel[3];

#ifndef SFML_STATIC
void window_init() {}
void window_show() {}
bool window_is_open() { return true; }
void window_delete() {}
void window_draw_point(double x, double y, bool color = false) {}
#endif

void cpu_alloc() {
	for (int i = 0; i < 3; i++) {
		_pos[i] = pos[i] = new double[AMOUNT];
		_vel[i] = vel[i] = new double[AMOUNT];
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

		if (grid[grid_pos[X]][grid_pos[Y]][grid_pos[Z]]) {
			i--;
			continue;
		}

		grid[grid_pos[X]][grid_pos[Y]][grid_pos[Z]] = true;
	}

	set_pos();
	set_vel();
}

void dump() {
	get_pos();
	get_vel();
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
		set_pos();
		set_vel();
	}
}

double deflect(double& p) {
	if (p < 0)
		p += (trunc((-p) / SIZE) + 1) * SIZE;
	if (p > SIZE)
		p -= trunc(p / SIZE) * SIZE;
	return p;
}

int main() {
	cpu_alloc();
	gpu_alloc();
	randomize();
	//load();
	window_init();
	
	while(window_is_open()) {
		for(int i = 0; i < SKIPS; i++)
			euler_step();
		for (int i = 0; i < AMOUNT; i++) {
			get_pos();
			window_draw_point(deflect(pos[X][i]) * OUTPUT_COEFF, deflect(pos[Y][i]) * OUTPUT_COEFF);
		}
		window_show();
		cout << get_energy() << endl;
	}
	window_delete();

	dump();
	gpu_dealloc();
	return 0;
}