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
			vel[j][i] = ((double)rand() / RAND_MAX - .5) * 2 * _sqrt(3 * K * T / get_properties(i).M);
		}

		if (grid[grid_pos[X]][grid_pos[Y]][grid_pos[Z]]) {
			i--;
			continue;
		}

		grid[grid_pos[X]][grid_pos[Y]][grid_pos[Z]] = true;
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

double deflect(double& p) {
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
#ifdef ENABLE_EM
	res += 9;
#endif
	return res;
}

long long ntime() {
	return chrono::duration_cast<chrono::nanoseconds>(chrono::system_clock::now().time_since_epoch()).count();
}

void flops_output(long long t0) {
	long long dt = (long long)ntime() - t0;
	cout << "dt = " << dt / 1000000 << " ms (" << SKIPS * AMOUNT * AMOUNT * flop() / dt << " GFlops)" << endl;
}

void output(vector<compute*>& to_cout, vector<compute*>& to_csv, string filename) {
	static ofstream out(filename);
	out.precision(15);
	for(int i = 0; i < to_csv.size(); i++) {
		to_csv[i]->calculate();
		to_csv[i]->output_csv(out, i == to_csv.size() - 1 ? "\n" : ",");
	}
	for(int i = 0; i < to_cout.size(); i++) {
		to_cout[i]->calculate();
		to_cout[i]->output_cout();
	}
}


bool interrupt = false;
int main(int argc, char* argv[], char* envp[]) {
	if (!(argc > 1 ? selectDevice(std::stoi(argv[1])) : selectDevice(0)))
		return -1;

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

		static vector<compute*> to_cout = { 
			new total_energy(),
			new temperature(),
			new total_pressure(),
			new virial_pressure() };

		static vector<compute*> to_csv = {
			new elapsed_time(),
			new potential_energy(),
			new kinetic_energy(),
			new total_energy(),
			new temperature(),
			new temperature_pressure(),
			new virial_pressure(),
			new total_pressure(),
			new tvm_du()
		};

		if (i <= NSTEPS / 2)
			total_time = 0;

		if (NSTEPS != -1) cout << i + 1 << "/" << NSTEPS << ": ";
		output(to_cout, i < NSTEPS / 2 ? vector<compute*>(0) : to_csv, "data/datadump_1280_10bar.csv");

		pull_values();		
		flops_output(t0);
		print_err(false);
	}
	window_delete();

	//dump("data/dump.dat");
	dealloc();
	return 0;
}
