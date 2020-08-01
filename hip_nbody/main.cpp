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

double SIZE[3] = {10, 10, 10};
int SCREEN_SIZE[2] = {500, 500}; 

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

void dump(string filename, int amount = AMOUNT) {
	pull_values();
	ofstream out(filename, ios::binary);
	for (int i = 0; i < 3; i++) {
		out.write((char*)pos[i], amount * sizeof(double));
		out.write((char*)vel[i], amount * sizeof(double));
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

double deflect(double p, double SIZE) {
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
void lmp_dump(string name);
void load_atom(string name);
void forces_dump(string name);

bool interrupt = false;
int main(int argc, char* argv[], char* envp[]) {
	alloc();
	load_atom("lmp.atom");
	SCREEN_SIZE[Y] = (int)(SCREEN_SIZE[X] * SIZE[Y] / SIZE[X]);
	init();
	push_values();
	window_init();
	pull_values();
	signal(SIGINT, [](int sig) { interrupt = true; });

	for(int i = 0; i != NSTEPS && window_is_open() && !interrupt; i++) {
		long long t0 = ntime();
		euler_steps(SKIPS);

	#ifdef SFML_STATIC
		double OUTPUT_COEFF = SCREEN_SIZE[X] / SIZE[X];
		for (int j = 0; j < AMOUNT; j++)
			window_draw_point(deflect(pos[X][j], SIZE[X]) * OUTPUT_COEFF, deflect(pos[Y][j], SIZE[Y]) * OUTPUT_COEFF);
		window_show();
	#endif

		
		if (NSTEPS != -1) cout << i + 1 << "/" << NSTEPS << ": ";

		static vector<compute*> to_cout = {
			new elapsed_time(),
			new total_energy(),
			new temperature(),
			new potential_energy(),
			new kinetic_energy()
			#ifdef ENABLE_SC
				,new sc_thermostat_dE()
			#endif
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
		//apply_andersen_thermostat(0.000);
		//flops_output(t0);
		cout << endl;
	}
	window_delete();
	lmp_dump("dump.lmp");
	forces_dump("omm.force");
	return 0;
}
