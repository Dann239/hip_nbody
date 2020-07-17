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
OpenMM::CustomNonbondedForce* lj; 
void alloc() {
	for(int i = 0; i < 3; i++) {
		pos[i] = new double[AMOUNT];
		vel[i] = new double[AMOUNT];
	}
	sys = new OpenMM::System();
	lj = new OpenMM::CustomNonbondedForce(
		"4*epsilon*((sigma/r)^12-(sigma/r)^6);"
		"sigma=0.5*(sigma1+sigma2);"
		"epsilon=sqrt(epsilon1*epsilon2);"
	);
	lj->addPerParticleParameter("sigma");
	lj->addPerParticleParameter("epsilon");
	lj->setNonbondedMethod(OpenMM::CustomNonbondedForce::NonbondedMethod::CutoffPeriodic);
	for(int i = 0; i < AMOUNT; i++) {
		sys->addParticle(1);
		lj->addParticle(vector<double>{1, 1});
	}
	sys->addForce(lj);
	
	verlet = new OpenMM::VerletIntegrator(TIME_STEP);
	context = new OpenMM::Context(*sys, *verlet);
	context->setPeriodicBoxVectors(
		OpenMM::Vec3(SIZE, 0, 0),
		OpenMM::Vec3(0, SIZE, 0),
		OpenMM::Vec3(0, 0, SIZE));
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
