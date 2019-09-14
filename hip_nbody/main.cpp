#include "kernel.h"
#include "window.h"
#include "properties.h"

#include <vector>
#include <iostream>
#include <fstream>
#include <time.h>
#include <math.h>
using namespace std;

constexpr double OUTPUT_COEFF = SCREEN_SIZE / SIZE;

static double* pos[3];
static double* vel[3];

double** _pos = pos;
double** _vel = vel;

#ifndef SFML_STATIC
void window_init() {}
void window_show() {}
bool window_is_open() { return true; }
void window_delete() {}
void window_draw_point(double x, double y, bool color = false) {}
#endif

void randomize() {
	constexpr int grid_size = (int)(_cbrt(AMOUNT) + 1);
	static bool grid[grid_size][grid_size][grid_size];
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

	push_values();
}

void dump() {
	pull_values();
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
		push_values();
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
	alloc();
	randomize();
	//load();
	
	window_init();
	
	while(window_is_open()) {
		long long t0 = clock();
		
		euler_steps(SKIPS);

#ifdef SFML_STATIC
		for (int i = 0; i < AMOUNT; i++)
			window_draw_point(deflect(pos[X][i]) * OUTPUT_COEFF, deflect(pos[Y][i]) * OUTPUT_COEFF, properties(i / BLOCK_SIZE).COLOUR);
		window_show();
#endif

		pull_values();
		cout << "mspf: " << ((long long)clock() - t0) * 1000 / CLOCKS_PER_SEC << "; e = " << total_energy << endl;
	}
	window_delete();

	dump();
	gpu_dealloc();
	return 0;
}
