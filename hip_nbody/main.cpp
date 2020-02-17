#include "kernel.h"
#include "window.h"
#include "properties.h"

#include <vector>
#include <iostream>
#include <fstream>
#include <time.h>
#include <math.h>
#include <chrono>
#include <csignal>
using namespace std;

bool interrupt = false;
void interrupter(int n) {
	interrupt = true;
}

constexpr long long flop() {
	int res = 14;
	#ifdef ENABLE_LJ
		res += 13;
	#endif
	#ifdef ENABLE_EM
		res += 9;
	#endif
	return res;
}

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

void dump() {
	pull_values();
	ofstream out("data/dump.dat", ios::binary);
	for (int i = 0; i < 3; i++) {
		out.write((char*)pos[i], AMOUNT * sizeof(double));
		out.write((char*)vel[i], AMOUNT * sizeof(double));
	}
	out.close();
}
void load() {
	ifstream in("data/dump_relaxed.dat", ios::binary);
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

long long ntime() {
	return chrono::duration_cast<chrono::nanoseconds>(chrono::system_clock::now().time_since_epoch()).count();
}

double potential_energy = 0;
double field_energy = 0;
double kinetic_energy_i = 0;
double kinetic_energy_e = 0;
double kinetic_energy = 0;
double dedv_pressure = 0;
double virial = 0;
double current = 0;
void energy_calc() {
	potential_energy = 0;
	field_energy = 0;
	dedv_pressure = 0;
	kinetic_energy_i = 0;
	kinetic_energy_e = 0;
	virial = 0;
	current = 0;
	int amount_i = 0, amount_e = 0;
	for (int i = 0; i < AMOUNT; i++) {
		field_energy -= get_properties(i).Q * pos[X][i] * E_EXT / AMOUNT;
		potential_energy += enrg[i] / AMOUNT;

		double dk = get_properties(i).M * (vel[X][i] * vel[X][i] + vel[Y][i] * vel[Y][i] + vel[Z][i] * vel[Z][i]) / 2;
		if(get_elem_type(i) == ELECTRON) {
			kinetic_energy_e += dk;
			amount_e++;
		}
		else {
			kinetic_energy_i += dk;
			amount_i++;
		}
		current += (N * get_properties(i).Q * vel[X][i]) / AMOUNT;
		dedv_pressure += dedv[i];
		virial += viri[i];
	}
	kinetic_energy = (kinetic_energy_e + kinetic_energy_i) / AMOUNT;
	if(amount_e) kinetic_energy_e /= amount_e;
	if(amount_i) kinetic_energy_i /= amount_i;
}

void datadump() {
	static ofstream out("data/datadump.csv");
	out.precision(30);
	out << (2. / 3. * kinetic_energy_i / K) << ','; //Ti
	out << (2. / 3. * kinetic_energy_e / K) << ','; //Te
	out << current << ','; //j
	out << virial << ','; //p_v
	out << (potential_energy + kinetic_energy + field_energy) / E; //E
	out << endl;
}

void output(long long t0) {
	cout.precision(6);
	cout << fixed << "E = " << ((potential_energy + kinetic_energy + field_energy) / E * 1e3) << " meV; ";
	cout.precision(3);
	cout << fixed << "T = " << (2. / 3. * kinetic_energy_i / K) << " K; ";
	cout << fixed << "Te = " << (2. / 3. * kinetic_energy_e / K) << " K; ";
	cout << fixed << "p_viri = " << virial * 1000 << " mPa; ";
	cout << fixed << "j = " << current / 1000 << " kA/m2; ";
	cout << "dt = " << ((long long)ntime() - t0) / 1000000 << " ms (" << (flop() * SKIPS * AMOUNT * AMOUNT) / ((long long)ntime() - t0) << " GFlops)" << endl;
}

int main(int argc, char* argv[], char* envp[]) {
	if (!(argc > 1 ? selectDevice(std::stoi(argv[1])) : selectDevice(0)))
		return -1;

	print_chars();

	alloc();
	randomize();
	load();

	window_init();

	force_energy_calc();
	pull_values();

	signal(SIGINT, interrupter);

	for(int i = 0; i != NSTEPS && window_is_open() && !interrupt; i++) {
		long long t0 = ntime();
		euler_steps(SKIPS);

	#ifdef SFML_STATIC
		constexpr double OUTPUT_COEFF = SCREEN_SIZE / SIZE;
		for (int i = 0; i < AMOUNT; i++)
			window_draw_point(deflect(pos[X][i]) * OUTPUT_COEFF, deflect(pos[Y][i]) * OUTPUT_COEFF, get_properties(i).COLOUR);
		window_show();
	#endif
		energy_calc();
		datadump();
		pull_values();
		output(t0);
		print_err(false);
	}
	window_delete();

	dump();
	dealloc();
	return 0;
}
