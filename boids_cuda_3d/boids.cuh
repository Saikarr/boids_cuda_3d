#ifndef BOIDS_H
#define BOIDS_H

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

#include <cmath>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define WIDTH 800
#define HEIGHT 600
#define DEPTH 600
#define MAX_SPEED 1.0f

struct BoidSoA {
	float* x;
	float* y;
	float* z;
	float* vx;
	float* vy;
	float* vz;
};

// CPU functions
void limit_speed_cpu(float& vx, float& vy, float& vz, float max_speed);
void update_boids_cpu(BoidSoA boids, int num_boids, float cohesion_weight, float alignment_weight, float separation_weight, float perception_radius, int i);

// Common functions
void initialize_boids(BoidSoA& boids, int num_boids);
int cuda_boid_initialize(BoidSoA& d_boids, BoidSoA h_boids, int num_boids);
void free_boids(BoidSoA& d_boids, BoidSoA& h_boids);
void free_all(BoidSoA& d_boids, BoidSoA& h_boids, GLFWwindow* window);
int copy_back(BoidSoA& h_boids, BoidSoA b_boids, int num_boids);

// CUDA functions
__device__ void limit_speed(float& vx, float& vy, float& vz, float max_speed);
__global__ void update_boids(BoidSoA boids, int num_boids, float cohesion_weight, float alignment_weight, float separation_weight, float perception_radius);

#endif // BOIDS_H