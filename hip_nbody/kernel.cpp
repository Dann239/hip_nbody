#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <memory>
#include "OpenMM.h"
#include "kernel.h"
#include "properties.h"


using namespace std;

double** pos = new double*[3];
double** vel = new double*[3];
double** acc = new double*[3];

double* enrg = new double[AMOUNT];
double* viri = new double[AMOUNT];
double* types = new double[AMOUNT];

OpenMM::System* sys;
OpenMM::VerletIntegrator* verlet;
OpenMM::Context* context;

constexpr double E = 1.602176634e-19;
constexpr double NA = 6.02214076e23;
constexpr double kJmol_per_eV = E * NA * 1e-3;
//constexpr double BOHR = 5.29177210903e-11 * 1e9;
//constexpr double HARTREE = 4.3597447222071e-18 * 1e-3 * NA;
constexpr double BOHR = 0.529 * 0.1;
constexpr double HARTREE = 27.2 * kJmol_per_eV;
using func1D = OpenMM::Continuous1DFunction;
using func2D = OpenMM::Continuous2DFunction;
using func3D = OpenMM::Continuous3DFunction;

double deflect(double p, double SIZE);
void lmp_dump(string name) {
	ofstream out(name);
	if(out.fail()) throw runtime_error("Cannot open .lmp file to write");
	out << endl;
	out << AMOUNT << " atoms" << endl;
	out << 2 << " atom types" << endl;
	out << 0 << " " << SIZE[X] * OpenMM::AngstromsPerNm << " xlo xhi" << endl;
	out << 0 << " " << SIZE[Y] * OpenMM::AngstromsPerNm << " ylo yhi" << endl;
	out << 0 << " " << SIZE[Z] * OpenMM::AngstromsPerNm << " zlo zhi" << endl;

	out << endl << "Atoms" << endl << endl;
	for(int i = 0; i < AMOUNT; i++)
		out << i + 1 << " " << types[i] << " " 
			<< deflect(pos[X][i], SIZE[X]) * OpenMM::AngstromsPerNm << " "
			<< deflect(pos[Y][i], SIZE[Y]) * OpenMM::AngstromsPerNm << " "
			<< deflect(pos[Z][i], SIZE[Z]) * OpenMM::AngstromsPerNm << endl;
}
void forces_dump(string name) {
	ofstream out(name);
	for(int i = 0; i < AMOUNT; i++)
		out << i + 1 << ' ' 
		<< acc[X][i] / kJmol_per_eV / OpenMM::AngstromsPerNm << ' ' 
		<< acc[Y][i] / kJmol_per_eV / OpenMM::AngstromsPerNm << ' ' 
		<< acc[Z][i] / kJmol_per_eV / OpenMM::AngstromsPerNm << endl;
}
void load_atom(string name) {
	ifstream in(name);
	if(in.fail()) throw runtime_error("Cannot open .atom file");
	in.ignore(0xFFFF, '\n');
	in.ignore(0xFFFF, '\n');
	in.ignore(0xFFFF, '\n');
	int amount;
	in >> amount;
	if(amount != AMOUNT) throw runtime_error("Amount in .atom file doesn't match the AMOUNT value");
	in.ignore(0xFFFF, '\n');
	in.ignore(0xFFFF, '\n');
	double low[3], high[3];
	for(int i = 0; i < 3; i++) in >> low[i] >> high[i];
	for(int i = 0; i < 3; i++) SIZE[i] = (high[i] - low[i]) * OpenMM::NmPerAngstrom;
	in.ignore(0xFFFF, '\n');
	in.ignore(0xFFFF, '\n');
	for(int i = 0; i < AMOUNT; i++) {
		int n, t;
		double p[3];
		in >> n >> t;
		types[n - 1] = t - 1;
		for(int j = 0; j < 3; j++) in >> p[j];
		for(int j = 0; j < 3; j++) pos[j][n - 1] = SIZE[j] * p[j];
		for(int j = 0; j < 3; j++) vel[j][n - 1] = 0;
	}
}
OpenMM::CustomGBForce* read_EAM_file(string filename, double& M) {
	ifstream in(filename);
	if(in.fail()) throw runtime_error("Cannot open .eam file");
	in.ignore(0xFFFF, '\n');
	string dummy;
	in >> dummy >> M >> dummy >> dummy;
	int Nrho, Nr;
	double drho, dr, cutoff;
	in >> Nrho >> drho >> Nr >> dr >> cutoff;
	vector<double> F_values(Nrho), Z_values(Nr), rho_values(Nr);
	for(int i = 0; i < Nrho; i++) in >> F_values[i];
	for(int i = 0; i < Nr; i++) in >> Z_values[i];
	for(int i = 0; i < Nr; i++) in >> rho_values[i];
	in.close();

	dr *= OpenMM::NmPerAngstrom;
	cutoff *= OpenMM::NmPerAngstrom;
	for(int i = 0; i < Nrho; i++) F_values[i] *= kJmol_per_eV;
	for(int i = 0; i < Nr; i++) Z_values[i] *= _sqrt(HARTREE * BOHR);

	func1D* F = new func1D(F_values, 0, drho*(Nrho - 1));
	func1D* Z = new func1D(Z_values, 0, dr*(Nr - 1));
	func1D* rho = new func1D(rho_values, 0, dr*(Nr - 1));

	auto eam = new OpenMM::CustomGBForce();

	eam->setNonbondedMethod(OpenMM::CustomGBForce::CutoffPeriodic);
	eam->setCutoffDistance(cutoff);

	eam->addTabulatedFunction("F", F);
	eam->addTabulatedFunction("Z", Z);
	eam->addTabulatedFunction("rho_f", rho);

	eam->addComputedValue("rho", "rho_f(r)", OpenMM::CustomGBForce::ComputationType::ParticlePair);
	eam->addEnergyTerm("F(rho)", OpenMM::CustomGBForce::ComputationType::SingleParticle);
	eam->addEnergyTerm("Z(r)*Z(r)/r", OpenMM::CustomGBForce::ComputationType::ParticlePair);

	return eam;
}

OpenMM::CustomGBForce* read_EAM_ALLOY_file(string filename, vector<double>& M, string select_one_elem = "none") {
	ifstream in(filename);
	if(in.fail()) throw runtime_error("Cannot open .eam.alloy file");
	in.ignore(0xFFFF, '\n');
	in.ignore(0xFFFF, '\n');
	in.ignore(0xFFFF, '\n');
	int n_elements;
	in >> n_elements;
	M.resize(n_elements);
	string dummy[n_elements][3];
	string elems[n_elements];
	for(int i = 0; i < n_elements; i++) in >> elems[i];

	int Nrho, Nr;
	double drho, dr, cutoff;
	in >> Nrho >> drho >> Nr >> dr >> cutoff;
	vector<vector<double> > F_values(n_elements, vector<double>(Nrho));
	vector<vector<double> > rho_values(n_elements, vector<double>(Nr));
	for(int i = 0; i < n_elements; i++) {
		in >> dummy[i][0] >> M[i] >> dummy[i][1] >> dummy[i][2];
		for(int j = 0; j < Nrho; j++) in >> F_values[i][j];
		for(int j = 0; j < Nr; j++) in >> rho_values[i][j];
	}
	vector<vector<vector<double>>> phi_values(n_elements,vector<vector<double>>(n_elements,vector<double>(Nr)));
	for(int i = 0; i < n_elements; i++)
		for(int j = 0; j <= i; j++)
			for(int k = 0; k < Nr; k++) {
				in >> phi_values[i][j][k];
				phi_values[j][i][k] = phi_values[i][j][k];
			}
	in.close();

	/*//////////////////////////////////
	ofstream out("AlO_own.eam.alloy");
	out.precision(16);
	out << scientific;

	out << "own config of AlO setfl file" << endl << endl << endl;
	out << n_elements << " ";
	for(int i = 0; i < n_elements; i++)
		out << elems[i] << " ";
	out << endl;
	out << Nrho << " " << drho << " " << Nr << " " << dr << " " << cutoff << endl;
	for(int i = 0; i < n_elements; i++) {
		out << dummy[i][0] << " " << M[i] << " " << dummy[i][1] << " " << dummy[i][2] << endl;
		for(int j = 0; j < Nrho; j++) out << F_values[i][j] << endl;
		for(int j = 0; j < Nr; j++) out << rho_values[i][j] << endl;
	}
	for(int i = 0; i < n_elements; i++)
		for(int j = 0; j <= i; j++)
			for(int k = 0; k < Nr; k++)
				out << 0 << endl;//phi_values[i][j][k] << endl;
	out.close();
	*////////////////////////////////////
		
	dr *= OpenMM::NmPerAngstrom;
	cutoff *= OpenMM::NmPerAngstrom;
	for(int i = 0; i < n_elements; i++)
		for(int j = 0; j < Nrho; j++)
			F_values[i][j] *= kJmol_per_eV;
	
	
	for(int i = 0; i < n_elements; i++)
		for(int j = 0; j < n_elements; j++)
			for(int k = 0; k < Nr; k++)
				phi_values[i][j][k] *= kJmol_per_eV * OpenMM::NmPerAngstrom;

	auto eam = new OpenMM::CustomGBForce();
	eam->setNonbondedMethod(OpenMM::CustomGBForce::CutoffPeriodic);
	eam->setCutoffDistance(cutoff);

	if(select_one_elem == "none") {
		vector<double> F_values_flat(n_elements * Nrho);	
		vector<double> rho_values_flat(n_elements * Nr);	
		vector<double> phi_values_flat(n_elements * n_elements * Nr);
		for(int i = 0; i < n_elements; i++) {
			for(int j = 0; j < Nrho; j++)
				F_values_flat[i + n_elements*j] = F_values[i][j];
			for(int j = 0; j < Nr; j++)
				rho_values_flat[i + n_elements*j] = rho_values[i][j];
			for(int j = 0; j < n_elements; j++)
				for(int k = 0; k < Nr; k++)
					phi_values_flat[i + n_elements*j + n_elements*n_elements*k] = phi_values[i][j][k];
		}
		func2D* F = new func2D(n_elements, Nrho, F_values_flat, 1, n_elements, 0, drho*(Nrho - 1));
		func2D* rho = new func2D(n_elements, Nr, rho_values_flat, 1, n_elements, 0, dr*(Nr - 1));
		func3D* phi = new func3D(n_elements, n_elements, Nr, phi_values_flat, 1, n_elements, 1, n_elements, 0, dr*(Nr - 1));

		eam->addPerParticleParameter("type");

		eam->addTabulatedFunction("F", F);
		eam->addTabulatedFunction("rho_f", rho);
		eam->addTabulatedFunction("phi", phi);

		eam->addComputedValue("rho", "rho_f(type2, r)", OpenMM::CustomGBForce::ComputationType::ParticlePair);
		eam->addEnergyTerm("F(type, rho)", OpenMM::CustomGBForce::ComputationType::SingleParticle);
		eam->addEnergyTerm("phi(type1, type2, r)/r", OpenMM::CustomGBForce::ParticlePair);
	}
	else {
		int desired_type = -1;
		for(int i = 0; i < n_elements; i++)
			if(elems[i] == select_one_elem)
				desired_type = i;
		if(desired_type == -1) throw runtime_error("Desired element not detected in .eam.alloy file");

		func1D* F = new func1D(F_values[desired_type], 0, drho*(Nrho - 1));
		func1D* rho = new func1D(rho_values[desired_type], 0, dr*(Nr - 1));
		func1D* phi = new func1D(phi_values[desired_type][desired_type], 0, dr*(Nr - 1));

		eam->addTabulatedFunction("F", F);
		eam->addTabulatedFunction("rho_f", rho);
		eam->addTabulatedFunction("phi", phi);

		eam->addComputedValue("rho", "rho_f(r)", OpenMM::CustomGBForce::ComputationType::ParticlePair);
		eam->addEnergyTerm("F(rho)", OpenMM::CustomGBForce::ComputationType::SingleParticle);
		eam->addEnergyTerm("phi(r)/r", OpenMM::CustomGBForce::ParticlePair);
	}
	return eam;
}

void alloc() {
	for(int i = 0; i < 3; i++) {
		pos[i] = new double[AMOUNT];
		vel[i] = new double[AMOUNT];
		acc[i] = new double[AMOUNT];
	}
}

void init() {
	sys = new OpenMM::System();
	sys->setDefaultPeriodicBoxVectors(
		OpenMM::Vec3(SIZE[X], 0, 0),
		OpenMM::Vec3(0, SIZE[Y], 0),
		OpenMM::Vec3(0, 0, SIZE[Z]));
	vector<double> M;
	
	auto eam = read_EAM_ALLOY_file("AlO.eam.alloy", M, "O");
	for(int i = 0; i < AMOUNT; i++) sys->addParticle(M[types[i]]);
	for(int i = 0; i < AMOUNT; i++) eam->addParticle();
	sys->addForce(eam);
	verlet = new OpenMM::VerletIntegrator(TIME_STEP);

	OpenMM::Platform::loadPluginsFromDirectory(OpenMM::Platform::getDefaultPluginsDirectory());
	context = new OpenMM::Context(*sys, *verlet,
		OpenMM::Platform::getPlatformByName("CUDA"),
		map<string, string> {{"Precision", "double"}});
	cout << "Using OpenMM platform " << context->getPlatform().getName().c_str() << endl;
}

double kenergy;
void pull_values() {
	auto state = context->getState(OpenMM::State::Positions | OpenMM::State::Velocities | OpenMM::State::Energy | OpenMM::State::Forces | OpenMM::State::Parameters);
	auto omm_pos = state.getPositions();
	auto omm_vel = state.getVelocities();
	auto omm_enrg = state.getPotentialEnergy();
	kenergy = state.getKineticEnergy();
	auto omm_acc = state.getForces();
	for(int i = 0; i < AMOUNT; i++) 
		for(int j = 0; j < 3; j++) {
			pos[j][i] = omm_pos[i][j];
			vel[j][i] = omm_vel[i][j];
			acc[j][i] = omm_acc[i][j];
			enrg[i] = omm_enrg / AMOUNT;
			viri[i] = 0;
		}
}
void push_values() {
	vector<OpenMM::Vec3> omm_pos(AMOUNT);
	vector<OpenMM::Vec3> omm_vel(AMOUNT);
	for(int i = 0; i < AMOUNT; i++) 
		for(int j = 0; j < 3; j++) {
			omm_pos[i][j] = pos[j][i];
			omm_vel[i][j] = vel[j][i];
		}
	context->setPositions(omm_pos);
	context->setVelocitiesToTemperature(300);
}

double total_time = 0;
void euler_steps(int steps) {
	total_time += TIME_STEP * SKIPS;
	verlet->step(SKIPS);
}
