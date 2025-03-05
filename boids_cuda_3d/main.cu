#include <stdlib.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "boids.cuh"
#include "render.h"

#define BLOCK_SIZE 1024
#define WIDTH 800
#define HEIGHT 600
#define DEPTH 600
#define MAX_SPEED 1.0f

int num_boids = 2048;
float perception_radius = 40.0f;
float cohesion_weight = 0.1f;
float alignment_weight = 0.15f;
float separation_weight = 0.2f;

void usage(const char* prog) {
	printf("Usage: %s <computation_method> <number_of_boids>\n"
		"<computation_method> - gpu or cpu\n<number_of_boids> - number of boids which will be displayed", prog);
}

int main(int argc, char* argv[]) {

	// Check the arguments
	if (argc != 3) usage(argv[0]);
	const char* computation_method = argv[1];
	const char* number_of_boids = argv[2];
	bool is_gpu = false;

	if (strcmp(computation_method, "gpu") != 0 && strcmp(computation_method, "cpu") != 0) {
		usage(argv[0]);
		exit(EXIT_FAILURE);
	}

	if (strcmp(computation_method, "gpu") == 0) is_gpu = true;

	if ((num_boids = atoi(number_of_boids)) <= 0) {
		usage(argv[0]);
		exit(EXIT_FAILURE);
	}

	// Initialize the boids
	BoidSoA h_boids;
	h_boids.x = new float[num_boids];
	h_boids.y = new float[num_boids];
	h_boids.z = new float[num_boids];
	h_boids.vx = new float[num_boids];
	h_boids.vy = new float[num_boids];
	h_boids.vz = new float[num_boids];

	initialize_boids(h_boids, num_boids);

	BoidSoA d_boids;
	if (cuda_boid_initialize(d_boids, h_boids, num_boids) == 1) {
		free_boids(d_boids, h_boids);
		exit(EXIT_FAILURE);
	}

	// Variables for the kernel
	dim3 block(BLOCK_SIZE);
	dim3 grid((num_boids + BLOCK_SIZE - 1) / BLOCK_SIZE);

	// Initialize GLFW and GLEW
	if (!glfwInit()) {
		std::cerr << "Failed to initialize GLFW" << std::endl;
		free_boids(d_boids, h_boids);
		return -1;
	}

	GLFWwindow* window = glfwCreateWindow(WIDTH, HEIGHT, "3D Boids Simulation", NULL, NULL);
	if (!window) {
		std::cerr << "Failed to create GLFW window" << std::endl;
		glfwTerminate();
		free_boids(d_boids, h_boids);
		return -1;
	}

	glfwMakeContextCurrent(window);
	glewExperimental = GL_TRUE;
	if (glewInit() != GLEW_OK) {
		std::cerr << "Failed to initialize GLEW" << std::endl;
		free_boids(d_boids, h_boids);
		return -1;
	}

	glEnable(GL_DEPTH_TEST);

	// Initialize ImGui
	IMGUI_CHECKVERSION();
	ImGui::CreateContext();
	ImGui_ImplGlfw_InitForOpenGL(window, true);
	ImGui_ImplOpenGL3_Init("#version 130");
	ImGui::StyleColorsDark();

	float camera_angle = 0.0f;
	float camera_distance = 800.0f;
	auto last_time = std::chrono::high_resolution_clock::now();
	int frame_count = 0;
	float fps = 0.0f;
	cudaError_t err;

	// Main loop
	while (!glfwWindowShouldClose(window)) {

		// Calculate the FPS
		auto current_time = std::chrono::high_resolution_clock::now();
		std::chrono::duration<float> elapsed = current_time - last_time;
		frame_count++;
		if (elapsed.count() >= 1.0f) {
			fps = frame_count / elapsed.count();
			frame_count = 0;
			last_time = current_time;
		}

		// Calculate the new positions of the boids
		if (is_gpu) {
			update_boids << <grid, block >> > (d_boids, num_boids, cohesion_weight, alignment_weight, separation_weight, perception_radius);
			err = cudaGetLastError();
			if (err != cudaSuccess) {
				std::cerr << "Failed to launch kernel: " << cudaGetErrorString(err) << std::endl;
				free_all(d_boids, h_boids, window);
				exit(EXIT_FAILURE);
			}

			err = cudaDeviceSynchronize(); // Wait for the kernel to finish
			if (err != cudaSuccess) {
				std::cerr << "cudaDeviceSynchronize failed: " << cudaGetErrorString(err) << std::endl;
				free_all(d_boids, h_boids, window);
				exit(EXIT_FAILURE);
			}

			// Copy the boids back to the host
			if (copy_back(h_boids, d_boids, num_boids) == 1) {
				free_all(d_boids, h_boids, window);
				exit(EXIT_FAILURE);
			}
		}
		else {
			for (int i = 0; i < num_boids; i++) {
				update_boids_cpu(h_boids, num_boids, cohesion_weight, alignment_weight, separation_weight, perception_radius, i);
			}
		}

		// Render the boids and the GUI
		render(window, h_boids, num_boids, fps, camera_angle, camera_distance);

		// Handle the movement of the camera
		if (glfwGetKey(window, GLFW_KEY_LEFT) == GLFW_PRESS) {
			camera_angle -= 0.02f;
		}
		if (glfwGetKey(window, GLFW_KEY_RIGHT) == GLFW_PRESS) {
			camera_angle += 0.02f;
		}
		if (glfwGetKey(window, GLFW_KEY_UP) == GLFW_PRESS) {
			camera_distance -= 10.0f;
		}
		if (glfwGetKey(window, GLFW_KEY_DOWN) == GLFW_PRESS) {
			camera_distance += 10.0f;
		}
	}

	// Free the memory
	free_all(d_boids, h_boids, window);
	return 0;
}
