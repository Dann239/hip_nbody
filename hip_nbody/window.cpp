#include "window.h"

#include <queue>
using namespace std;

#ifdef SFML_STATIC

#include "SFML/Graphics.hpp"
using namespace sf;

RenderWindow *window;

void window_init() {
	ContextSettings settings;
	window = new RenderWindow;
	settings.antialiasingLevel = 8;
	window->create(sf::VideoMode(SCREEN_SIZE, SCREEN_SIZE), "Runge CUDA", sf::Style::Default, settings);
	window->setFramerateLimit(120);
}
void window_show() {
	window->display();
	window->setSize(Vector2u(SCREEN_SIZE, SCREEN_SIZE));
	window->clear();

	static queue<Event> events;
	static sf::Event event;
	
	while (window->pollEvent(event))
		events.push(event);

	while (events.size() > 0) {
		switch (events.front().type) {
		case Event::EventType::Closed:
			window->close();
		default:
			break;
		}
		events.pop();
	}
}
bool window_is_open() {
	return window->isOpen();
}
void window_delete() {
	delete window;
}
void window_draw_point(double x, double y, int colour) {
	
	const float r = SCREEN_SIZE * 4e-3f;
	static CircleShape point;
	point.setRadius(r);
	point.setPointCount(6);
	point.setFillColor(Color(Uint32(colour)));
	point.setPosition(Vector2f((float)x - r, (float)y - r));
	window->draw(point);
}

#endif
