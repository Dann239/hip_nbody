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

potential_energy::potential_energy() : average(enrg), compute("Ep = ", 1, "; ") {}

kinetic_energy::kinetic_energy() : compute("Ek = ", 1, "; ") {}
extern double kenergy;
double kinetic_energy::calculate() {
	return kenergy;
}

total_energy::total_energy() : compute("E = ", 1, "; ", 6) {}
double total_energy::calculate() {
	return value = kinetic_energy().calculate() + potential_energy().calculate();
}

temperature::temperature() : compute("T = ", 1, "; ") {}
double temperature::calculate() {
	return value = (2. / 3.) * _cbrt(2) * kinetic_energy().calculate();
}

elapsed_time::elapsed_time() : compute("t = ", 1, "; ") {}
double elapsed_time::calculate() {
	return value = total_time;
}

double complete_state::calculate() { return 0; }
void complete_state::output_csv(std::ofstream& stream, std::string ending) {
	stream << AMOUNT << '\n' << elapsed_time().calculate() << '\n';
	for(int i = 0; i < AMOUNT; i++) {
		for(int j = 0; j < 3; j++)
			stream << pos[j][i] << ' ';
		for(int j = 0; j < 3; j++)
			stream << vel[j][i] << ' ';
		stream << '\n';
	}
}

lindemann::lindemann() : compute("<du2> = ", 1, "; ") {
	for(int i = 0; i < AMOUNT; i++)
		for(int j = 0; j < 3; j++)
			pos_init[j].push_back(pos[j][i]);
}
double lindemann::calculate() {
	double res = 0;
	for(int i = 0; i < AMOUNT; i++)
		for(int j = 0; j < 3; j++)
			res += (pos[j][i] - pos_init[j][i]) * (pos[j][i] - pos_init[j][i]) / AMOUNT / 3;
	return value = res;
}
