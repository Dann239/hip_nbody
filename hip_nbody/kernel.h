#pragma once

extern double** pos;
extern double** vel;
extern double** acc;

extern double*& enrg;
extern double*& viri;
extern double*& tvm;

extern double total_time;
extern double total_sc_thermostat_dE;

void alloc();
void dealloc();

void pull_values();
void push_values();

void euler_steps(int steps);
void force_energy_calc();

void print_err(bool force = true);
void print_chars();

bool selectDevice(int deviceIndex);