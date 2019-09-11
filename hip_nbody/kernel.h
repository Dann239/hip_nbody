#pragma once

void gpu_alloc();
void gpu_dealloc();

void get_pos(double* _pos[3]);
void get_vel(double* _vel[3]);

void set_pos(double* _pos[3]);
void set_vel(double* _vel[3]);

double get_energy();
void euler_step();

void print_err();