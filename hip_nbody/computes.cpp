#include <iostream>
#include "computes.h"
#include "properties.h"
#include "kernel.h"

compute::compute(std::string _prefix, double _coeff, std::string _postfix, int _precision) :
	cout_prefix(_prefix), cout_coeff(_coeff), cout_postfix(_postfix), cout_precision(_precision) {}
void compute::output_csv(std::ofstream& stream, std::string ending) {
	stream << value << ending;
}
void compute::output_cout() {
	std::cout.precision(cout_precision);
	std::cout << std::fixed << cout_prefix << value * cout_coeff << cout_postfix;
}

average::average(double*& _source) : source(&_source) {}
double average::calculate() {
	value = 0;
	for (int i = 0; i < AMOUNT; i++)
		value += (*source)[i] / AMOUNT;
	return value;
}

total::total(double*& _source) : source(&_source) {}
double total::calculate() {
	value = 0;
	for (int i = 0; i < AMOUNT; i++)
		value += (*source)[i];
	return value;
}

potential_energy::potential_energy() : average(enrg), compute("Ep = ", 1000 / E, " meV; ") {}

kinetic_energy::kinetic_energy() : compute("Ek = ", 1000 / E, " meV; ") {}
double kinetic_energy::calculate() {
	value = 0;
	for (int i = 0; i < AMOUNT; i++)
		value += get_properties(i).M * (
			vel[X][i] * vel[X][i] +
			vel[Y][i] * vel[Y][i] +
			vel[Z][i] * vel[Z][i]) / 2. / AMOUNT;
	return value;
}

total_energy::total_energy() : compute("E = ", 1000 / E, " meV; ", 6) {}
double total_energy::calculate() {
	return value = kinetic_energy().calculate() + potential_energy().calculate();
}

temperature::temperature() : compute("T = ", 1, " K; ") {}
double temperature::calculate() {
	return value = (2. / 3. / K) * kinetic_energy().calculate();
}

temperature_pressure::temperature_pressure() : compute("Pt = ", 1, " Pa; ") {}
double temperature_pressure::calculate() {
	return value = N * K * temperature().calculate();
}

virial_pressure::virial_pressure() : total(viri), compute("Pv = ", 1, " Pa; ", 5) {}

total_pressure::total_pressure() : compute("P = ", 1, " Pa; ") {}
double total_pressure::calculate() {
	return value = temperature_pressure().calculate() + virial_pressure().calculate();
}

tvm_du::tvm_du() : total(tvm), compute("dUtvm/dV = ", -1. / (V * 3. * ALPHA) * 1, " Pa; ", 5) {}

elapsed_time::elapsed_time() : compute("t = ", 1e9, " ns; ") {}
double elapsed_time::calculate() {
	return value = total_time;
}