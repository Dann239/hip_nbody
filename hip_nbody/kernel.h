#pragma once

extern double** pos;
extern double** vel;

extern double*& enrg;
extern double*& viri;
extern double*& tvm;

extern double total_time;

void alloc();
void dealloc();

void pull_values();
void push_values();

void euler_steps(int steps);
void force_energy_calc();

void print_err(bool force = true);
void print_chars();

bool selectDevice(int deviceIndex);