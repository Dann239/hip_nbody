#pragma once

//#define ENABLE_LJ
#define ENABLE_PB
//#define ENABLE_EAM
//#define ENABLE_SC

constexpr int AMOUNT = 2048; //2047 == 23 * 89

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

constexpr double T = 10 / _cbrt(2);
constexpr double N = 1.0 * _sqrt(2);

constexpr double V = AMOUNT / N;
constexpr double SIZE = _cbrt(V);

constexpr const char* OUTPUT_FILENAME = "data/FCC_LJ.xyz";
constexpr double TIME_STEP = 1e-5;
constexpr int SKIPS = 100;
constexpr int NSTEPS = -1;
constexpr double Z0 = 12;
constexpr double BETA = 2, A = .5;
constexpr double M = 1;


constexpr int MEM_LEN = AMOUNT * sizeof(double);

enum XYZ {X = 0, Y = 1, Z = 2};
