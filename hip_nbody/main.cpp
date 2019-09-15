#include "kernel.h"
#include "window.h"
#include "properties.h"

#include <vector>
#include <iostream>
#include <fstream>
#include <time.h>
#include <math.h>
using namespace std;

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
			pos[j][i] = (grid_pos[j] + (double)rand() / RAND_MAX / 2) * SIZE / grid_size;
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

long long properties::get_colour(int block) {
	for (int i = 1; i <= ELEMS_NUM; i++)
		if (block / (double)GRID_SIZE <= ELEMS_DIVISIONS[i])
			return properties(ELEMS_TYPES[i - 1]).COLOUR;
	return properties(ERROR).COLOUR;;
}

int main() {
	print_chars();
	//return 0;

	alloc();
	randomize();
	//load();

	window_init();
	force_energy_calc();

	//while(window_is_open()) {
	for (int i = 0; i < 5; i++) {
		long long t0 = clock();
		euler_steps(SKIPS);

	#ifdef SFML_STATIC
		constexpr double OUTPUT_COEFF = SCREEN_SIZE / SIZE;
		for (int i = 0; i < AMOUNT; i++)
			window_draw_point(deflect(pos[X][i]) * OUTPUT_COEFF, deflect(pos[Y][i]) * OUTPUT_COEFF, properties::get_colour(i / BLOCK_SIZE));
		window_show();
	#endif

		pull_values();
		cout << "mspf: " << ((long long)clock() - t0) * 1000 / CLOCKS_PER_SEC << "; e = " << total_energy << endl;
		print_err(false);
	}
	window_delete();

	dump();
	dealloc();
	return 0;
}
