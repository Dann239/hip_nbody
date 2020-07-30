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

OpenMM::Context* context;
OpenMM::VerletIntegrator* verlet;
OpenMM::System* sys;
OpenMM::NonbondedForce* klj;
OpenMM::CustomNonbondedForce* lj;
OpenMM::CustomGBForce* eam;
double M = 1;
constexpr double E = 1.602176634e-19;
constexpr double NA = 6.02214076e23;
constexpr double kJmol_per_eV = E * NA * 1e-3;
//constexpr double BOHR = 5.29177210903e-11 * 1e9;
//constexpr double HARTREE = 4.3597447222071e-18 * 1e-3 * NA;
constexpr double BOHR = 0.529 * 0.1;
constexpr double HARTREE = 27.2 * kJmol_per_eV;
using func1D = OpenMM::Continuous1DFunction;

double deflect(double p, double SIZE);
void lmp_dump(string name) {
	ofstream out(name);
	out << endl;
	out << AMOUNT << " atoms" << endl;
	out << 1 << " atom types" << endl;
	out << 0 << " " << SIZE[X] * OpenMM::AngstromsPerNm << " xlo xhi" << endl;
	out << 0 << " " << SIZE[Y] * OpenMM::AngstromsPerNm << " ylo yhi" << endl;
	out << 0 << " " << SIZE[Z] * OpenMM::AngstromsPerNm << " zlo zhi" << endl;

	out << endl << "Atoms" << endl << endl;
	for(int i = 0; i < AMOUNT; i++)
		out << i + 1 << " " << 1 << " " 
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
	in.ignore(0xFFFF, '\n');
	in.ignore(0xFFFF, '\n');
	in.ignore(0xFFFF, '\n');
	int amount;
	in >> amount;
	if(amount != AMOUNT) {
		cerr << "WRONG AMOUNT SET!!!" << endl;
		return;
	}
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
		for(int j = 0; j < 3; j++) in >> p[j];
		for(int j = 0; j < 3; j++) pos[j][n - 1] = SIZE[j] * p[j];

		for(int j = 0; j < 3; j++) vel[j][n - 1] = 0;
	}
}

tuple<func1D*, func1D*, func1D*> read_EAM_file(string filename, double& M, double& cutoff) {
	ifstream in(filename);
	in.ignore(0xFFFF, '\n');
	string dummy;
	in >> dummy >> M >> dummy >> dummy;
	int Nrho, Nr;
	double drho, dr;
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

	return make_tuple(
		new func1D(F_values, 0, drho*(Nrho - 1)),
		new func1D(Z_values, 0, dr*(Nr - 1)),
		new func1D(rho_values, 0, dr*(Nr - 1)));
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

	for(int i = 0; i < AMOUNT; i++) sys->addParticle(M);
	
	//CustomGBForce + Continuous1DFunction
	//lammps eam setfl

	double cutoff;
	func1D *F, *Z, *rho;
	tie(F, Z, rho) = read_EAM_file("Ni_u3.eam", M, cutoff);
	
	eam = new OpenMM::CustomGBForce();
	eam->setNonbondedMethod(OpenMM::CustomGBForce::CutoffPeriodic);
	eam->addTabulatedFunction("F", F);
	eam->addTabulatedFunction("Z", Z);
	eam->addTabulatedFunction("rho_f", rho);
	eam->addComputedValue("rho", "rho_f(r)", OpenMM::CustomGBForce::ComputationType::ParticlePairNoExclusions);
	eam->addEnergyTerm("F(rho)", OpenMM::CustomGBForce::ComputationType::SingleParticle);
	eam->addEnergyTerm("Z(r)*Z(r)/r", OpenMM::CustomGBForce::ComputationType::ParticlePairNoExclusions);
	eam->setCutoffDistance(cutoff);
	for(int i = 0; i < AMOUNT; i++) eam->addParticle();
	sys->addForce(eam);
	verlet = new OpenMM::VerletIntegrator(TIME_STEP);

	OpenMM::Platform::loadPluginsFromDirectory(OpenMM::Platform::getDefaultPluginsDirectory());
	context = new OpenMM::Context(*sys, *verlet,
		OpenMM::Platform::getPlatformByName("CUDA"),
		map<string, string> {{"Precision", "double"}});
	cout << "Using OpenMM platform " << context->getPlatform().getName().c_str() << endl;
}


void pull_values() {
	auto state = context->getState(OpenMM::State::Positions | OpenMM::State::Velocities | OpenMM::State::Energy | OpenMM::State::Forces | OpenMM::State::Parameters);
	auto omm_pos = state.getPositions();
	auto omm_vel = state.getVelocities();
	auto omm_enrg = state.getPotentialEnergy();
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
