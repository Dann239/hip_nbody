#pragma once

extern double** pos;
extern double** vel;
extern double** acc;

extern double*& energy;
extern double*& dedv;

extern double total_energy, potential_energy, kinetic_energy;

void alloc();
void dealloc();

void pull_values();
void push_values();

void euler_steps(int steps);
void force_energy_calc();

void print_err(bool force = true);
void print_chars();