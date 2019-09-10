#pragma once

void gpu_alloc();
void gpu_dealloc();

void get_pos(double* _pos[3]);
void get_vel(double* _vel[3]);

void set_pos(double* _pos[3]);
void set_vel(double* _vel[3]);

void print_err();

constexpr double _cbrt(double a) {
	if (a < 0)
		return -_cbrt(-a);
	if (a < 1)
		return 1 / _cbrt(1 / a);
	double l = 0, r = a;
	while (1) {
		double m = (l + r) / 2;
		if (m == l || m == r)
			return m;
		if ((double)m * m * m > (double)a)
			r = m;
		else
			l = m;
	}
}
constexpr double _sqrt(double a) {
	if (a < 0)
		return -_sqrt(-a);
	if (a < 1)
		return 1 / _sqrt(1 / a);
	double l = 0, r = a;
	while (1) {
		double m = (l + r) / 2;
		if (m == l || m == r)
			return m;
		if ((double)m * m > (double)a)
			r = m;
		else
			l = m;
	}
}


constexpr int GRID_SIZE = 1 << 7;
constexpr int BLOCK_SIZE = 1 << 7;
constexpr int AMOUNT = GRID_SIZE * BLOCK_SIZE;

constexpr int BUTCHER_SIZE = 15;

constexpr double PI = 3.14159265359;
constexpr double K = 1.380649e-23;
constexpr double NA = 6.022141e23;
constexpr double R = K * NA;

constexpr double SIGMA = 0.3405e-9;
constexpr double M = 0.040 / NA;
constexpr double EPSILON = 0.1198 * K;

constexpr double T = 300;
constexpr double P = 1e6;
constexpr double N = P / (K * T);//25.7444e27;

constexpr double V = _sqrt(3 * K * T / M);

constexpr double SIZE = _cbrt(AMOUNT / N);
constexpr int SCREEN_SIZE = 500;
constexpr double OUTPUT_COEFF = SCREEN_SIZE / SIZE;

constexpr double TIME_STEP = 5e-15;
constexpr double RENDER_STEP = TIME_STEP * 50;
constexpr double TOTAL_TIME = RENDER_STEP * 10;