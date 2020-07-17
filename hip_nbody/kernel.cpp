#include "OpenMM.h"
#include "kernel.h"
#include "properties.h"

using namespace std;

double** pos = new double*[3];
double** vel = new double*[3];

double* enrg = new double[AMOUNT];
double* viri = new double[AMOUNT];

OpenMM::Context* context;
OpenMM::VerletIntegrator* verlet;
OpenMM::System* sys;
OpenMM::NonbondedForce* klj;
OpenMM::CustomNonbondedForce* lj;
OpenMM::CustomGBForce* eam;

void alloc() {
	for(int i = 0; i < 3; i++) {
		pos[i] = new double[AMOUNT];
		vel[i] = new double[AMOUNT];
	}
	sys = new OpenMM::System();
	for(int i = 0; i < AMOUNT; i++) sys->addParticle(M);

	klj = new OpenMM::NonbondedForce();
	klj->setNonbondedMethod(OpenMM::NonbondedForce::NonbondedMethod::CutoffPeriodic);
	for(int i = 0; i < AMOUNT; i++) klj->addParticle(0, 1, 1);
	sys->addForce(klj);
	
	/*
	lj = new OpenMM::CustomNonbondedForce(
		"4*epsilon*((sigma/r)^12-(sigma/r)^6)");
	lj->addGlobalParameter("sigma", 1);
	lj->addGlobalParameter("epsilon", 1);
	lj->setNonbondedMethod(OpenMM::CustomNonbondedForce::NonbondedMethod::CutoffPeriodic);
	for(int i = 0; i < AMOUNT; i++) lj->addParticle();
	sys->addForce(lj);
	*/


	eam = new OpenMM::CustomGBForce();
	eam->addGlobalParameter("A", A);
	eam->addGlobalParameter("beta", BETA);
	eam->addGlobalParameter("Z0", Z0);
	eam->addComputedValue("rho", "exp(-beta*(r-1))/Z0", OpenMM::CustomGBForce::ComputationType::ParticlePairNoExclusions);
	eam->addComputedValue("F", "A*Z0/2*rho*(log(rho)-1)", OpenMM::CustomGBForce::ComputationType::SingleParticle);
	eam->addEnergyTerm("F", OpenMM::CustomGBForce::ComputationType::SingleParticle);
	eam->addEnergyTerm("-A/Z0*exp(-beta*(r-1))*(-beta*(r-1)-1)", OpenMM::CustomGBForce::ComputationType::ParticlePairNoExclusions);
	eam->setNonbondedMethod(OpenMM::CustomGBForce::CutoffPeriodic);
	for(int i = 0; i < AMOUNT; i++) eam->addParticle();
	sys->addForce(eam);

	verlet = new OpenMM::VerletIntegrator(TIME_STEP);

	OpenMM::Platform::loadPluginsFromDirectory(OpenMM::Platform::getDefaultPluginsDirectory());
	context = new OpenMM::Context(*sys, *verlet,
		OpenMM::Platform::getPlatformByName("CUDA"),
		map<string, string> {{"Precision", "mixed"}});
	context->setPeriodicBoxVectors(
		OpenMM::Vec3(SIZE, 0, 0),
		OpenMM::Vec3(0, SIZE, 0),
		OpenMM::Vec3(0, 0, SIZE));

	printf( "Using OpenMM platform %s\n", 
        context->getPlatform().getName().c_str() );
}


void pull_values() {
	auto state = context->getState(OpenMM::State::Positions | OpenMM::State::Velocities | OpenMM::State::Energy);
	auto omm_pos = state.getPositions();
	auto omm_vel = state.getVelocities();
	auto omm_enrg = state.getPotentialEnergy();
	for(int i = 0; i < AMOUNT; i++) 
		for(int j = 0; j < 3; j++) {
			pos[j][i] = omm_pos[i][j];
			vel[j][i] = omm_vel[i][j];
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
	context->setVelocities(omm_vel);
}

double total_time = 0;
void euler_steps(int steps) {
	total_time += TIME_STEP * SKIPS;
	verlet->step(SKIPS);
}
