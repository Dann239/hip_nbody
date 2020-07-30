#pragma once

extern double** pos;
extern double** vel;
extern double** acc;

extern double* enrg;
extern double* viri;

extern double total_time;

void alloc();
void init();

void pull_values();
void push_values();

void euler_steps(int steps);
void force_energy_calc();
