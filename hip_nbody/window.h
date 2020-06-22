#pragma once

constexpr int SCREEN_SIZE = 500;

void window_init();
void window_show();
bool window_is_open();
void window_delete();
void window_draw_point(double x, double y, unsigned int colour = 0);