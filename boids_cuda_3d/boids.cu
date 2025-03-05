#include "boids.cuh"
#include <cstdlib>
#include <iostream>

// Function to limit the speed of a boid
__device__ void limit_speed(float& vx, float& vy, float& vz, float max_speed) {
	float speed = sqrtf(vx * vx + vy * vy + vz * vz);
	if (speed > max_speed) {
		vx = (vx / speed) * max_speed;
		vy = (vy / speed) * max_speed;
		vz = (vz / speed) * max_speed;
	}
}

__global__ void update_boids(BoidSoA boids, int num_boids, float cohesion_weight, float alignment_weight, float separation_weight, float perception_radius) {
	// Get the index of the boid
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= num_boids) return;

	float current_x = boids.x[i];
	float current_y = boids.y[i];
	float current_z = boids.z[i];
	float current_vx = boids.vx[i];
	float current_vy = boids.vy[i];
	float current_vz = boids.vz[i];

	float cohesion_x = 0.0f, cohesion_y = 0.0f, cohesion_z = 0.0f;
	float alignment_x = 0.0f, alignment_y = 0.0f, alignment_z = 0.0f;
	float separation_x = 0.0f, separation_y = 0.0f, separation_z = 0.0f;
	int neighbor_count = 0;

	// Loop through all boids
	for (int j = 0; j < num_boids; ++j) {
		if (i == j) continue;

		float dx = boids.x[j] - current_x;
		float dy = boids.y[j] - current_y;
		float dz = boids.z[j] - current_z;
		float distance = sqrtf(dx * dx + dy * dy + dz * dz);

		// Check if the boid is within the perception radius
		if (distance < perception_radius) {
			// Cohesion
			cohesion_x += boids.x[j];
			cohesion_y += boids.y[j];
			cohesion_z += boids.z[j];

			// Alignment
			alignment_x += boids.vx[j];
			alignment_y += boids.vy[j];
			alignment_z += boids.vz[j];

			// Separation
			if (distance > 0) {
				separation_x -= dx / distance;
				separation_y -= dy / distance;
				separation_z -= dz / distance;
			}

			// Increment the neighbor count
			neighbor_count++;
		}
	}

	if (neighbor_count > 0) {
		// Cohesion
		cohesion_x /= neighbor_count;
		cohesion_y /= neighbor_count;
		cohesion_z /= neighbor_count;
		current_vx += (cohesion_x - current_x) * cohesion_weight;
		current_vy += (cohesion_y - current_y) * cohesion_weight;
		current_vz += (cohesion_z - current_z) * cohesion_weight;

		// Alignment
		alignment_x /= neighbor_count;
		alignment_y /= neighbor_count;
		alignment_z /= neighbor_count;
		current_vx += alignment_x * alignment_weight;
		current_vy += alignment_y * alignment_weight;
		current_vz += alignment_z * alignment_weight;

		// Separation
		current_vx += separation_x * separation_weight;
		current_vy += separation_y * separation_weight;
		current_vz += separation_z * separation_weight;
	}

	// Limit speed
	limit_speed(current_vx, current_vy, current_vz, MAX_SPEED);

	// Update position
	current_x += current_vx;
	current_y += current_vy;
	current_z += current_vz;

	// Move to the other side of the screen if the boid goes out of bounds
	if (current_x < 0) current_x += WIDTH;
	if (current_x >= WIDTH) current_x -= WIDTH;
	if (current_y < 0) current_y += HEIGHT;
	if (current_y >= HEIGHT) current_y -= HEIGHT;
	if (current_z < 0) current_z += DEPTH;
	if (current_z >= DEPTH) current_z -= DEPTH;

	// Update the boid
	boids.x[i] = current_x;
	boids.y[i] = current_y;
	boids.z[i] = current_z;
	boids.vx[i] = current_vx;
	boids.vy[i] = current_vy;
	boids.vz[i] = current_vz;
}

// Function to limit the speed of a boid
void limit_speed_cpu(float& vx, float& vy, float& vz, float max_speed) {
	float speed = sqrtf(vx * vx + vy * vy + vz * vz);
	if (speed > max_speed) {
		vx = (vx / speed) * max_speed;
		vy = (vy / speed) * max_speed;
		vz = (vz / speed) * max_speed;
	}
}

void update_boids_cpu(BoidSoA boids, int num_boids, float cohesion_weight, float alignment_weight, float separation_weight, float perception_radius, int i) {

	float current_x = boids.x[i];
	float current_y = boids.y[i];
	float current_z = boids.z[i];
	float current_vx = boids.vx[i];
	float current_vy = boids.vy[i];
	float current_vz = boids.vz[i];

	float cohesion_x = 0.0f, cohesion_y = 0.0f, cohesion_z = 0.0f;
	float alignment_x = 0.0f, alignment_y = 0.0f, alignment_z = 0.0f;
	float separation_x = 0.0f, separation_y = 0.0f, separation_z = 0.0f;
	int neighbor_count = 0;

	// Loop through all boids
	for (int j = 0; j < num_boids; ++j) {
		if (i == j) continue;

		float dx = current_x - boids.x[j];
		float dy = current_y - boids.y[j];
		float dz = current_z - boids.z[j];
		float distance = sqrtf(dx * dx + dy * dy + dz * dz);

		// Check if the boid is within the perception radius
		if (distance < perception_radius) {
			// Cohesion
			cohesion_x += boids.x[j];
			cohesion_y += boids.y[j];
			cohesion_z += boids.z[j];

			// Alignment
			alignment_x += boids.vx[j];
			alignment_y += boids.vy[j];
			alignment_z += boids.vz[j];

			// Separation
			if (distance > 0) {
				separation_x += dx / distance;
				separation_y += dy / distance;
				separation_z += dz / distance;
			}

			// Increment the neighbor count
			neighbor_count++;
		}
	}

	if (neighbor_count > 0) {
		// Cohesion
		cohesion_x /= neighbor_count;
		cohesion_y /= neighbor_count;
		cohesion_z /= neighbor_count;
		current_vx += (cohesion_x - current_x) * cohesion_weight;
		current_vy += (cohesion_y - current_y) * cohesion_weight;
		current_vz += (cohesion_z - current_z) * cohesion_weight;

		// Alignment
		alignment_x /= neighbor_count;
		alignment_y /= neighbor_count;
		alignment_z /= neighbor_count;
		current_vx += alignment_x * alignment_weight;
		current_vy += alignment_y * alignment_weight;
		current_vz += alignment_z * alignment_weight;

		// Separation
		current_vx += separation_x * separation_weight;
		current_vy += separation_y * separation_weight;
		current_vz += separation_z * separation_weight;
	}

	// Limit speed
	limit_speed_cpu(current_vx, current_vy, current_vz, MAX_SPEED);

	// Update position
	current_x += current_vx;
	current_y += current_vy;
	current_z += current_vz;

	// Move to the other side of the screen if the boid goes out of bounds
	if (current_x < 0) current_x += WIDTH;
	if (current_x >= WIDTH) current_x -= WIDTH;
	if (current_y < 0) current_y += HEIGHT;
	if (current_y >= HEIGHT) current_y -= HEIGHT;
	if (current_z < 0) current_z += DEPTH;
	if (current_z >= DEPTH) current_z -= DEPTH;

	// Update the boid
	boids.x[i] = current_x;
	boids.y[i] = current_y;
	boids.z[i] = current_z;
	boids.vx[i] = current_vx;
	boids.vy[i] = current_vy;
	boids.vz[i] = current_vz;
}

// Function to initialize the host boids
void initialize_boids(BoidSoA& boids, int num_boids) {
	for (int i = 0; i < num_boids; ++i) {
		boids.x[i] = rand() % WIDTH;
		boids.y[i] = rand() % HEIGHT;
		boids.z[i] = rand() % DEPTH;
		boids.vx[i] = (rand() / RAND_MAX * 2.0 - 1.0) * MAX_SPEED;
		boids.vy[i] = (rand() / RAND_MAX * 2.0 - 1.0) * MAX_SPEED;
		boids.vz[i] = (rand() / RAND_MAX * 2.0 - 1.0) * MAX_SPEED;
	}
}

// Function to initialize the device boids
int cuda_boid_initialize(BoidSoA& d_boids, BoidSoA h_boids, int num_boids) {
	cudaError_t err;

	// Allocate device memory
	err = cudaMalloc(&d_boids.x, num_boids * sizeof(float));
	if (err != cudaSuccess) {
		std::cerr << "Failed to allocate device memory for d_boids.x: " << cudaGetErrorString(err) << std::endl;
		return 1;
	}
	err = cudaMalloc(&d_boids.y, num_boids * sizeof(float));
	if (err != cudaSuccess) {
		std::cerr << "Failed to allocate device memory for d_boids.y: " << cudaGetErrorString(err) << std::endl;
		return 1;
	}
	err = cudaMalloc(&d_boids.z, num_boids * sizeof(float));
	if (err != cudaSuccess) {
		std::cerr << "Failed to allocate device memory for d_boids.z: " << cudaGetErrorString(err) << std::endl;
		return 1;
	}
	err = cudaMalloc(&d_boids.vx, num_boids * sizeof(float));
	if (err != cudaSuccess) {
		std::cerr << "Failed to allocate device memory for d_boids.vx: " << cudaGetErrorString(err) << std::endl;
		return 1;
	}
	err = cudaMalloc(&d_boids.vy, num_boids * sizeof(float));
	if (err != cudaSuccess) {
		std::cerr << "Failed to allocate device memory for d_boids.vy: " << cudaGetErrorString(err) << std::endl;
		return 1;
	}
	err = cudaMalloc(&d_boids.vz, num_boids * sizeof(float));
	if (err != cudaSuccess) {
		std::cerr << "Failed to allocate device memory for d_boids.vz: " << cudaGetErrorString(err) << std::endl;
		return 1;
	}

	// Copy memory to device
	err = cudaMemcpy(d_boids.x, h_boids.x, num_boids * sizeof(float), cudaMemcpyHostToDevice);
	if (err != cudaSuccess) {
		std::cerr << "Failed to copy memory to device for d_boids.x: " << cudaGetErrorString(err) << std::endl;
		return 1;
	}
	err = cudaMemcpy(d_boids.y, h_boids.y, num_boids * sizeof(float), cudaMemcpyHostToDevice);
	if (err != cudaSuccess) {
		std::cerr << "Failed to copy memory to device for d_boids.y: " << cudaGetErrorString(err) << std::endl;
		return 1;
	}
	err = cudaMemcpy(d_boids.z, h_boids.z, num_boids * sizeof(float), cudaMemcpyHostToDevice);
	if (err != cudaSuccess) {
		std::cerr << "Failed to copy memory to device for d_boids.z: " << cudaGetErrorString(err) << std::endl;
		return 1;
	}
	err = cudaMemcpy(d_boids.vx, h_boids.vx, num_boids * sizeof(float), cudaMemcpyHostToDevice);
	if (err != cudaSuccess) {
		std::cerr << "Failed to copy memory to device for d_boids.vx: " << cudaGetErrorString(err) << std::endl;
		return 1;
	}
	err = cudaMemcpy(d_boids.vy, h_boids.vy, num_boids * sizeof(float), cudaMemcpyHostToDevice);
	if (err != cudaSuccess) {
		std::cerr << "Failed to copy memory to device for d_boids.vy: " << cudaGetErrorString(err) << std::endl;
		return 1;
	}
	err = cudaMemcpy(d_boids.vz, h_boids.vz, num_boids * sizeof(float), cudaMemcpyHostToDevice);
	if (err != cudaSuccess) {
		std::cerr << "Failed to copy memory to device for d_boids.vz: " << cudaGetErrorString(err) << std::endl;
		return 1;
	}
	return 0;
}

// Function to free the boids
void free_boids(BoidSoA& d_boids, BoidSoA& h_boids) {
	cudaFree(d_boids.x);
	cudaFree(d_boids.y);
	cudaFree(d_boids.z);
	cudaFree(d_boids.vx);
	cudaFree(d_boids.vy);
	cudaFree(d_boids.vz);

	delete[] h_boids.x;
	delete[] h_boids.y;
	delete[] h_boids.z;
	delete[] h_boids.vx;
	delete[] h_boids.vy;
	delete[] h_boids.vz;
}

// Function to free all resources
void free_all(BoidSoA& d_boids, BoidSoA& h_boids, GLFWwindow* window) {
	free_boids(d_boids, h_boids);
	ImGui_ImplOpenGL3_Shutdown();
	ImGui_ImplGlfw_Shutdown();
	ImGui::DestroyContext();
	glfwDestroyWindow(window);
	glfwTerminate();
}

// Function to copy the boids back to the host
int copy_back(BoidSoA& h_boids, BoidSoA b_boids, int num_boids) {
	cudaError_t err;

	err = cudaMemcpy(h_boids.x, b_boids.x, num_boids * sizeof(float), cudaMemcpyDeviceToHost);
	if (err != cudaSuccess) {
		std::cerr << "Failed to copy memory to host for h_boids.x: " << cudaGetErrorString(err) << std::endl;
		return 1;
	}
	err = cudaMemcpy(h_boids.y, b_boids.y, num_boids * sizeof(float), cudaMemcpyDeviceToHost);
	if (err != cudaSuccess) {
		std::cerr << "Failed to copy memory to host for h_boids.y: " << cudaGetErrorString(err) << std::endl;
		return 1;
	}
	err = cudaMemcpy(h_boids.z, b_boids.z, num_boids * sizeof(float), cudaMemcpyDeviceToHost);
	if (err != cudaSuccess) {
		std::cerr << "Failed to copy memory to host for h_boids.z: " << cudaGetErrorString(err) << std::endl;
		return 1;
	}
	err = cudaMemcpy(h_boids.vx, b_boids.vx, num_boids * sizeof(float), cudaMemcpyDeviceToHost);
	if (err != cudaSuccess) {
		std::cerr << "Failed to copy memory to host for h_boids.vx: " << cudaGetErrorString(err) << std::endl;
		return 1;
	}
	err = cudaMemcpy(h_boids.vy, b_boids.vy, num_boids * sizeof(float), cudaMemcpyDeviceToHost);
	if (err != cudaSuccess) {
		std::cerr << "Failed to copy memory to host for h_boids.vy: " << cudaGetErrorString(err) << std::endl;
		return 1;
	}
	err = cudaMemcpy(h_boids.vz, b_boids.vz, num_boids * sizeof(float), cudaMemcpyDeviceToHost);
	if (err != cudaSuccess) {
		std::cerr << "Failed to copy memory to host for h_boids.vz: " << cudaGetErrorString(err) << std::endl;
		return 1;
	}
	return 0;
}