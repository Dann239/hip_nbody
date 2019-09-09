#pragma once

constexpr int GRID_SIZE = 1 << 2;
constexpr int BLOCK_SIZE = 1 << 2;
constexpr int AMOUNT = GRID_SIZE * BLOCK_SIZE;

void gpu_alloc();
void gpu_dealloc();

void get_pos(double* _p[3]);
void get_vel(double* _v[3]);

void set_pos(double* _p[3]);
void set_vel(double* _v[3]);

void print_err();