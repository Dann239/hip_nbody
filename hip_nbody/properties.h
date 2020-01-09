#pragma once

#define ENABLE_EM
//#define ENABLE_LJ

constexpr int BLOCK_SIZE = 128; //optimal is 128 * N for nvidia, 256 * N for amd
constexpr int GRID_SIZE = 10; //optimal is SMM_count * M
constexpr int AMOUNT = GRID_SIZE * BLOCK_SIZE;

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

constexpr double PI = 3.14159265359;
constexpr double K = 1.380649e-23;
constexpr double NA = 6.022141e23;
constexpr double R = K * NA;
constexpr double EPSILON0 = 8.85418781762e-12;
constexpr double MU0 = 4e-7 * PI;
constexpr double E = 1.60217662e-19;

constexpr double T = 1000;
constexpr double P = 1;
constexpr double N = P / (K * T);

constexpr double V = AMOUNT / N;
constexpr double SIZE = _cbrt(V);

constexpr double R0 = 1e-8;
constexpr double TIME_STEP = 1e-14;
constexpr int SKIPS = 250;
constexpr int NSTEPS = -1;

constexpr int MEM_LEN = AMOUNT * sizeof(double);

enum XYZ {X = 0, Y, Z};
enum ELEMS {ASTATINE, HELIUM, ELECTRON, PROTON, ERROR};

constexpr int ELEMS_NUM = 2;
constexpr double ELEMS_DIVISIONS[ELEMS_NUM + 1] = { 0, 0.49999, 1 };
constexpr ELEMS ELEMS_TYPES[ELEMS_NUM] = {HELIUM, ASTATINE};

struct properties {
	double SIGMA, EPSILON, M, Q;
	unsigned int COLOUR;
	double divisions[ELEMS_NUM + 1];
	void set_properties(ELEMS type) {
		switch (type)
		{
		case ASTATINE:
			SIGMA = 0.3405e-9;
			M = 0.040 / NA;
			EPSILON = 119.8 * K;
			Q = E;
			COLOUR = 0x0000FFFF;
			break;
		case HELIUM:
			SIGMA = 0.263e-9;
			M = 0.004 / NA;
			EPSILON = 6.03 * K;
			Q = -E;
			COLOUR = 0xFF0000FF;
			break;
		case PROTON:
			SIGMA = 0.37e-9;
			M = 0.001 / NA;
			EPSILON = 30 * K;
			Q = E;
			COLOUR = 0xFF7F00FF;
			break;
		case ELECTRON:
			SIGMA = 1e-9;
			M = 9.1e-31;
			EPSILON = 0;
			Q = -E;
			COLOUR = 0xFFFFFFFF;
			break;
		default:
			SIGMA = EPSILON = M = Q = -1;
			COLOUR = 0x777777FF;
		}
		for(int i = 0; i <= ELEMS_NUM; i++)
			divisions[i] = ELEMS_DIVISIONS[i];
	}
	properties(ELEMS e) {
		set_properties(e);
	}
};

properties get_properties(int num);
