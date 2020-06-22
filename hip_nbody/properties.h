#pragma once

#define ENABLE_LJ
#define ENABLE_PB
#define ENABLE_EAM

constexpr int BLOCK_SIZE = 128; //optimal is 128 * N for nvidia, 256 * N for amd
constexpr int GRID_SIZE = 16; //optimal is SMM_count * M
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

constexpr double PI = 3.1415926535897932;

constexpr double RC = _sqrt(_cbrt(2));

constexpr double T = 0.1 / _cbrt(2);
constexpr double N = 1 * _sqrt(2);

constexpr double V = AMOUNT / N;
constexpr double SIZE = _cbrt(V);

constexpr const char* OUTPUT_FILENAME = "data/FCC_LJ.xyz";
constexpr double TIME_STEP = 1e-3;
constexpr int SKIPS = 100;
constexpr int NSTEPS = 300;
constexpr double Z0 = 12;
constexpr double BETA = 0.5, A = 1.5;


constexpr int MEM_LEN = AMOUNT * sizeof(double);

enum XYZ {X = 0, Y = 1, Z = 2};
enum ELEMS {LJ_PARTICLE, ERROR};

constexpr int ELEMS_NUM = 1;
constexpr double ELEMS_DIVISIONS[ELEMS_NUM + 1] = { 0, 1 };
constexpr ELEMS ELEMS_TYPES[ELEMS_NUM] = {LJ_PARTICLE};

struct properties {
	double SIGMA, EPSILON, M;
	unsigned int COLOUR;
	double divisions[ELEMS_NUM + 1];
	void set_properties(ELEMS type) {
		switch (type)
		{

		case LJ_PARTICLE:
			SIGMA = 1;
			M = 1;
			EPSILON = 1;
			COLOUR = 0xFF0000FF;
			break;
		default:
			SIGMA = EPSILON = M = -1;
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
ELEMS get_elem_type(int num);
