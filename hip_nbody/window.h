#pragma once

extern int SCREEN_SIZE[2];

void window_init();
void window_show();
bool window_is_open();
void window_delete();
void window_draw_point(double x, double y, unsigned int colour = 0xFF0000FF);