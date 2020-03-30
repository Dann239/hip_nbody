#pragma once
#include <fstream>
#include <string>

class compute {
private:
	std::string cout_prefix, cout_postfix;
	double cout_coeff;
	int cout_precision;
public:
	compute(std::string _prefix = "UNDEFINED", double _coeff = 0, std::string _postfix = "FORMAT", int _precision = 3);
	double value = 0;
	virtual double calculate() = 0;
	virtual void output_cout();
	virtual void output_csv(std::ofstream& stream, std::string ending);
};

class average : public virtual compute {
private:
	double** source;
public:
	average(double*& _source);
	double calculate() override final;
};

class total : public virtual compute {
private:
	double** source;
public:
	total(double*& _source);
	double calculate() override final;
};

class potential_energy : public average {
public:
	potential_energy();
};

class kinetic_energy : public compute {
public:
	kinetic_energy();
	double calculate() override;
};

class total_energy : public compute {
public:
	total_energy();
	double calculate() override;
};

class temperature : public compute {
public:
	temperature();
	double calculate() override;
};

class temperature_pressure : public compute {
public:
	temperature_pressure();
	double calculate() override;
};

class virial_pressure : public total {
public:
	virial_pressure();
};

class total_pressure : public compute {
public:
	total_pressure();
	double calculate() override;
};

class tvm_du : public total {
public:
	tvm_du();
};

class elapsed_time : public compute {
public:
	elapsed_time();
	double calculate() override;
};

class complete_state : public compute {
public:
	double calculate() override;
	void output_csv(std::ofstream& stream, std::string ending) override;
};