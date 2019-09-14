#pragma once

extern double* pos[3];
extern double* vel[3];
extern double total_energy;

void alloc();
void dealloc();

void pull_values();
void push_values();

void euler_steps(int steps);
void force_energy_calc();

void print_err(bool force = true);
void print_chars();
