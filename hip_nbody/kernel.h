#pragma once

extern double* _pos[3];
extern double* _vel[3];

void gpu_alloc();
void gpu_dealloc();

void get_pos();
void get_vel();

void set_pos();
void set_vel();

double get_energy();
void euler_step();

void print_err();