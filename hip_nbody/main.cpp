#include "kernel.h"
#include "window.h"
#include "properties.h"
#include "computes.h"

#include <vector>
#include <iostream>
#include <fstream>
#include <time.h>
#include <math.h>
#include <chrono>
#include <csignal>
#include <string>
using namespace std;

void randomize() {
	constexpr double cell_size = SIZE / LATTICE_STEP_COUNT;

	constexpr double a[3][3] = {
		{0, cell_size / 2., cell_size / 2.},
		{cell_size / 2., 0, cell_size / 2.},
		{cell_size / 2., cell_size / 2., 0}
	};

	for (int i = 0; i < LATTICE_STEP_COUNT; i++)
		for (int j = 0; j < LATTICE_STEP_COUNT; j++)
			for (int k = 0; k < LATTICE_STEP_COUNT; k++) {
				int offset = 4 * (k + j * LATTICE_STEP_COUNT + i * LATTICE_STEP_COUNT * LATTICE_STEP_COUNT);

				for (int dim = 0; dim < 3; dim++)
					for (int num = 0; num < 4; num++)
						vel[dim][offset + num] = 10;

				pos[X][offset + 0] = cell_size * i;  
				pos[Y][offset + 0] = cell_size * j;
				pos[Z][offset + 0] = cell_size * k;

				for (int dim = 0; dim < 3; dim++)
					for (int num = 0; num < 3; num++)
							pos[dim][offset + num + 1] = pos[dim][offset + 0] + a[num][dim];
			}

	push_values();
}

void dump(string filename) {
	pull_values();
	ofstream out(filename, ios::binary);
	for (int i = 0; i < 3; i++) {
		out.write((char*)pos[i], AMOUNT * sizeof(double));
		out.write((char*)vel[i], AMOUNT * sizeof(double));
	}
	out.close();
}
void load(string filename) {
	ifstream in(filename, ios::binary);
	if (!in.fail()) {
		for (int i = 0; i < 3; i++) {
			in.read((char*)pos[i], AMOUNT * sizeof(double));
			in.read((char*)vel[i], AMOUNT * sizeof(double));
		}
		push_values();
	}
	in.close();
}

double deflect(double p) {
	if (p < 0)
		p += (trunc((-p) / SIZE) + 1) * SIZE;
	if (p > SIZE)
		p -= trunc(p / SIZE) * SIZE;
	return p;
}

long long flop() {
	int res = 14;
#ifdef ENABLE_LJ
	res += 13;
#endif
	return res;
}

long long ntime() {
	return chrono::duration_cast<chrono::nanoseconds>(chrono::system_clock::now().time_since_epoch()).count();
}

void flops_output(long long t0) {
	long long dt = (long long)ntime() - t0;
	cout << "dt = " << dt / 1000000 << " ms (" << flop() * SKIPS * AMOUNT * AMOUNT / dt << " GFlops)" << endl;
}

void output_csv(const vector<compute*>& to_csv, ofstream& out) {
	if (to_csv.size()) {
		out.precision(15);
		for (int i = 0; i < to_csv.size(); i++) {
			to_csv[i]->calculate();
			to_csv[i]->output_csv(out, i == to_csv.size() - 1 ? "\n" : ",");
		}
	}
}
void output_cout(const vector<compute*>& to_cout) {
	for(int i = 0; i < to_cout.size(); i++) {
		to_cout[i]->calculate();
		to_cout[i]->output_cout();
	}
}

bool interrupt = false;
int main(int argc, char* argv[], char* envp[]) {
#ifndef __HCC__
	if (!(argc > 1 ? selectDevice(std::stoi(argv[1])) : selectDevice(0)))
		return -1;
#endif
	print_chars();

	alloc();
	randomize();
	//load("data/dump.dat");

	window_init();

	force_energy_calc();
	pull_values();

	signal(SIGINT, [](int sig) { interrupt = true; });

	for(int i = 0; i != NSTEPS && window_is_open() && !interrupt; i++) {
		long long t0 = ntime();
		euler_steps(SKIPS);

	#ifdef SFML_STATIC
		constexpr double OUTPUT_COEFF = SCREEN_SIZE / SIZE;
		for (int j = 0; j < AMOUNT; j++)
			window_draw_point(deflect(pos[X][j]) * OUTPUT_COEFF, deflect(pos[Y][j]) * OUTPUT_COEFF, get_properties(j).COLOUR);
		window_show();
	#endif

		
		if (NSTEPS != -1) cout << i + 1 << "/" << NSTEPS << ": ";

		static vector<compute*> to_cout = {
			new total_energy(),
			new temperature() };
		output_cout(to_cout);
		static string output_filename = OUTPUT_FILENAME;
		static ofstream out_csv(output_filename);
		static vector<compute*> to_csv = {
			new complete_state()
		};
		output_csv(to_csv, out_csv);
		
		pull_values();
		flops_output(t0);
		print_err(false);
	}
	window_delete();

	//dump("data/dump.dat");
	dealloc();
	return 0;
}
