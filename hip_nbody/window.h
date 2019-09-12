#pragma once

constexpr int SCREEN_SIZE = 500;

#ifdef SFML_STATIC
void window_init();
void window_show();
bool window_is_open();
void window_delete();
void window_draw_point(double x, double y, int colour = 0);
#endif
