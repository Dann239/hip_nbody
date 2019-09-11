#pragma once

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

constexpr int GRID_SIZE = 1 << 5;
constexpr int BLOCK_SIZE = 1 << 5;
constexpr int AMOUNT = GRID_SIZE * BLOCK_SIZE;

constexpr int BUTCHER_SIZE = 15;

constexpr double PI = 3.14159265359;
constexpr double K = 1.380649e-23;
constexpr double NA = 6.022141e23;
constexpr double R = K * NA;
constexpr double EPSILON0 = 8.85418781762e-12;
constexpr double MU0 = 4e-7 * PI;

constexpr double SIGMA = 0.3405e-9;
constexpr double EPSILON = 0.1198 * K;
constexpr double M = 0.040 / NA;
constexpr double Q = 1.60217662e-19;

constexpr double T = 300;
constexpr double P = 1e8;
constexpr double N = P / (K * T);

constexpr double V = _sqrt(3 * K * T / M);

constexpr double SIZE = _cbrt(AMOUNT / N);

constexpr double TIME_STEP = 1e-15;
constexpr double SKIPS = 5;