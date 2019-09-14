#pragma once

extern double** _pos;
extern double** _vel;
extern double total_energy;

void alloc();
void gpu_dealloc();

void push_values();
void pull_values();

void euler_steps(int steps);

void print_err();