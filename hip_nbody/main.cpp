#include "kernel.h"
#include "window.h"
#include "properties.h"
#include "computes.h"

#include <vector>
#include <array>
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
	if(T == 0)
		return 0;
	double basement = 1e-20;
	double floor_size = 0.00001;
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

void apply_andersen_thermostat(double T) {
	int pnum = rand() % AMOUNT;
	for(int i = 0; i < 3; i++)
		vel[i][pnum] = get_maxwell_speed(T, get_properties(pnum).M);
	push_values();
}

double total_sc_thermostat_dE = 0;
void apply_sc_thermostat() {
	double T_current = (new temperature())->calculate();
	double alpha = sqrt(T / T_current);

	for(int pnum = 0; pnum < AMOUNT; pnum++)
		for(int i = 0; i < 3; i++)
			vel[i][pnum] *= alpha;

	total_sc_thermostat_dE += (T - T_current) / ((2. / 3.) * _cbrt(2));
	push_values();
}

bool randomize_lattice(vector<array<double,3> > vecs) {
	int step_count = round(cbrt(AMOUNT / vecs.size()));
	if(vecs.size() * pow(step_count, 3) != AMOUNT) {
		cerr << "Invalid LATTICE_STEP_COUNT" << endl;
		return false;
	}
	double cell_size = SIZE / step_count;


	for (int i = 0; i < step_count; i++)
		for (int j = 0; j < step_count; j++)
			for (int k = 0; k < step_count; k++) {
				int offset = vecs.size() * (k + j * step_count + i * step_count * step_count);

				for (int dim = 0; dim < 3; dim++)
					for (int num = 0; num < vecs.size(); num++)
						vel[dim][offset + num] = get_maxwell_speed(T, get_properties(offset + num).M);

				pos[X][offset] = cell_size * i;
				pos[Y][offset] = cell_size * j;
				pos[Z][offset] = cell_size * k;

				for (int dim = 0; dim < 3; dim++)
					for (int num = 0; num < vecs.size(); num++)
						pos[dim][offset + num] = pos[dim][offset + 0] + cell_size * vecs[num][dim];
			}

	push_values();
	return true;
}

bool randomize_fcc() {
	return randomize_lattice(vector<array<double,3> >{
		{0,  0,  0},
		{.5, .5, 0},
		{.5, 0, .5},
		{0, .5, .5}});
}

bool randomize_bcc() {
	return randomize_lattice(vector<array<double,3> >{
		{0,  0,  0},
		{.5, .5, .5}});
}

bool randomize_sc() {
	return randomize_lattice(vector<array<double,3> >{
		{0,  0,  0}});
}

bool randomize_dc() {
	return randomize_lattice(vector<array<double,3> >{
		{0,  0,  0},
		{0,  .5,  .5},
		{.5,  0,  .5},
		{.5,  .5,  0},
		{.75,  .75,  .75},
		{.75,  .25,  .25},
		{.25,  .75,  .25},
		{.25,  .25,  .75}
		});
}

void randomize_default() {
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
			vel[j][i] = get_maxwell_speed(T, get_properties(i).M);
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
	cout << "dt = " << dt / 1000000 << " ms (" << flop() * SKIPS * AMOUNT * AMOUNT / dt << " GFlops)";
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
	if (!(argc > 1 ? selectDevice(std::stoi(argv[1])) : selectDevice(0))) {
		cout << "Could not select the device: ";
		print_err(true);
		return -1;
	}
#endif
	print_chars();

	alloc();
	//randomize_default();
	if (!randomize_fcc())
		return 0;
	
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
			new elapsed_time(),
			new total_energy(),
			new temperature(),
			new total_pressure(),
			new potential_energy(),
			new kinetic_energy()
		};
		output_cout(to_cout);
		
		//static string xyz_filename = OUTPUT_FILENAME;
		//static ofstream out_xyz(xyz_filename);
		//static vector<compute*> to_xyz = {new complete_state()};
		//output_csv(to_xyz, out_xyz);
		
		
		//static string csv_filename = "data/energy_example.csv";
		//static ofstream out_csv(csv_filename);
		//static vector<compute*> to_csv = to_cout;
		//output_csv(to_csv, out_csv);
		
		
		pull_values();
		#ifdef ENABLE_SC
			apply_sc_thermostat();
		#endif
		//apply_andersen_thermostat(0.000);

		//flops_output(t0);
		cout << endl;
		print_err(false);
	}
	window_delete();

	dealloc();
	return 0;
}
