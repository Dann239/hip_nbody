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

double uniform_rand() {
	return (double)rand() / RAND_MAX;
}
double maxwell(double v, double T, double m) {
	return sqrt(m / (2*PI*T)) * exp(-m*v*v / (2*T));
}
double inv_maxwell(double p, double T, double m) {
	return sqrt(2*T*log(sqrt(m / (2*PI*T)) / p) / m);
}
double get_maxwell_speed(double T, double m = 1) {
	double basement = 1e-20;
	double floor_size = 0.0001;
	double roof = maxwell(0, T, m);

	vector<double> ziggurat;
	double next_floor = basement;
	do {
		ziggurat.push_back(next_floor);
		next_floor += inv_maxwell(ziggurat.back(), T, m) / floor_size;
	} while(next_floor < roof);
	ziggurat.push_back(roof);
	
	double floor = rand() % (ziggurat.size() - 1);
	double vel_max = inv_maxwell(ziggurat[floor], T, m);
	double vel_next = inv_maxwell(ziggurat[floor + 1], T, m);

	double vel_new;
	do {
		vel_new = uniform_rand() * vel_max;
	} while(vel_new > vel_next && ziggurat[floor] + uniform_rand() * (ziggurat[floor+1] - ziggurat[floor]) > maxwell(vel_new, T, m));

	if(rand() & 1)
		vel_new = -vel_new;
	return vel_new;
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
			vel[j][i] = get_maxwell_speed(T, get_properties(i).M);//(uniform_rand() * 2 - 1) * _sqrt(3 * T / get_properties(i).M);
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
	//cout << get_maxwell_speed(1) << endl;
	//return 0;
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
