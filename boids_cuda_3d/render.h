#define GLEW_STATIC
#include <glew.h>
#include <GLFW/glfw3.h>
#include <glm.hpp>
#include <gtc/matrix_transform.hpp>
#include <gtc/type_ptr.hpp>
#include <GL/glu.h>

#include "imgui.h"
#include "backends/imgui_impl_glfw.h"
#include "backends/imgui_impl_opengl3.h"

#include "boids.cuh"

// Functions for rendering the boids and the GUI
void render_boids(const BoidSoA& boids, int num_boids);
void render_gui(float fps);
void render(GLFWwindow* window, const BoidSoA& h_boids, int num_boids, float fps, float camera_angle, float camera_distance);