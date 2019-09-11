#pragma once

constexpr int SCREEN_SIZE = 500;

#ifdef SFML_STATIC
void window_init();
void window_show();
bool window_is_open();
void window_delete();
void window_draw_point(double x, double y, bool color = false);
#else
void window_init() {}
void window_show() {}
bool window_is_open() { return true; }
void window_delete() {}
void window_draw_point(double x, double y, bool color = false) {}
#endif
