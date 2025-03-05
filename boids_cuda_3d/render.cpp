#include "render.h"

extern float cohesion_weight;
extern float alignment_weight;
extern float separation_weight;
extern float perception_radius;

void render_boids(const BoidSoA& boids, int num_boids) {
	for (int i = 0; i < num_boids; ++i) {
		float size = 12.0f;
		float height = 15.0f;

		// Compute the normalized direction vector
		float length = sqrtf(boids.vx[i] * boids.vx[i] + boids.vy[i] * boids.vy[i] + boids.vz[i] * boids.vz[i]);
		if (length == 0.0f) continue; // Skip if velocity is zero to avoid division by zero
		float dirX = boids.vx[i] / length;
		float dirY = boids.vy[i] / length;
		float dirZ = boids.vz[i] / length;

		// Compute rotation axis (cross product of reference axis (0, 0, 1) and direction)
		float refX = 0.0f, refY = 0.0f, refZ = 1.0f; // Reference axis (z-axis)
		float axisX = refY * dirZ - refZ * dirY;
		float axisY = refZ * dirX - refX * dirZ;
		float axisZ = refX * dirY - refY * dirX;

		// Compute rotation angle (dot product and arccos)
		float dot = refX * dirX + refY * dirY + refZ * dirZ;
		float angle = acosf(dot) * 180.0f / 3.14159f; // Convert to degrees

		glColor3f(1.0f, 0.0f, 0.0f);

		glPushMatrix();

		// Translate to the boid's position
		glTranslatef(boids.x[i], boids.y[i], boids.z[i]);

		// Apply rotation to align with velocity
		if (length > 0.0f) { // Avoid undefined rotation when velocity is zero
			glRotatef(angle, axisX, axisY, axisZ);
		}

		// Draw the cone
		glBegin(GL_TRIANGLES);

		// Base of the cone
		for (int j = 0; j < 360; j += 30) {
			float theta = j * 3.14159f / 180.0f;
			float nextTheta = (j + 30) * 3.14159f / 180.0f;
			glVertex3f(0.0f, 0.0f, 0.0f);  // Center of the base
			glVertex3f(cosf(theta) * size / 3, sinf(theta) * size / 3, 0.0f);  // Edge of the base
			glVertex3f(cosf(nextTheta) * size / 3, sinf(nextTheta) * size / 3, 0.0f);  // Next edge
		}

		// Sides of the cone
		for (int j = 0; j < 360; j += 30) {
			float theta = j * 3.14159f / 180.0f;
			float nextTheta = (j + 30) * 3.14159f / 180.0f;
			glVertex3f(0.0f, 0.0f, height);  // Tip of the cone
			glVertex3f(cosf(theta) * size / 3, sinf(theta) * size / 3, 0.0f);  // Edge of the base
			glVertex3f(cosf(nextTheta) * size / 3, sinf(nextTheta) * size / 3, 0.0f);  // Next edge
		}

		glEnd();

		glPopMatrix();
	}
}

// Render the GUI (sliders and text)
void render_gui(float fps) {
	ImGui::Begin("Boid Weights");
	ImGui::SliderFloat("Cohesion", &cohesion_weight, 0.0f, 1.0f);
	ImGui::SliderFloat("Alignment", &alignment_weight, 0.0f, 1.0f);
	ImGui::SliderFloat("Separation", &separation_weight, 0.0f, 1.0f);
	ImGui::SliderFloat("Perception radius", &perception_radius, 10.0f, 100.0f);
	ImGui::Text("FPS: %.1f", fps);
	ImGui::End();
}

void render(GLFWwindow* window, const BoidSoA& h_boids, int num_boids, float fps, float camera_angle, float camera_distance) {
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	// Set up the camera
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluPerspective(45.0, (double)WIDTH / (double)HEIGHT, 1.0, 3000.0);

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	gluLookAt(WIDTH / 2 + camera_distance * cos(camera_angle), HEIGHT / 2, DEPTH / 2 + camera_distance * sin(camera_angle), WIDTH / 2, HEIGHT / 2, DEPTH / 2, 0, 1, 0);

	// Draw boids
	render_boids(h_boids, num_boids);

	// Draw bounding box  
	glColor3f(1.0f, 1.0f, 1.0f);
	glBegin(GL_LINE_LOOP);
	glVertex3f(0, 0, 0);
	glVertex3f(WIDTH, 0, 0);
	glVertex3f(WIDTH, HEIGHT, 0);
	glVertex3f(0, HEIGHT, 0);
	glEnd();

	glBegin(GL_LINE_LOOP);
	glVertex3f(0, 0, DEPTH);
	glVertex3f(WIDTH, 0, DEPTH);
	glVertex3f(WIDTH, HEIGHT, DEPTH);
	glVertex3f(0, HEIGHT, DEPTH);
	glEnd();

	glBegin(GL_LINES);
	glVertex3f(0, 0, 0);
	glVertex3f(0, 0, DEPTH);
	glVertex3f(WIDTH, 0, 0);
	glVertex3f(WIDTH, 0, DEPTH);
	glVertex3f(WIDTH, HEIGHT, 0);
	glVertex3f(WIDTH, HEIGHT, DEPTH);
	glVertex3f(0, HEIGHT, 0);
	glVertex3f(0, HEIGHT, DEPTH);
	glEnd();

	// Render GUI
	ImGui_ImplOpenGL3_NewFrame();
	ImGui_ImplGlfw_NewFrame();
	ImGui::NewFrame();

	render_gui(fps);

	ImGui::Render();
	ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

	glfwSwapBuffers(window);
	glfwPollEvents();
}