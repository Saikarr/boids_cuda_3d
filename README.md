# 3D CUDA Boids

This project implements Craig Reynolds' Boids algorithm for simulating flocking behavior in 3D space, with acceleration using NVIDIA CUDA. The simulation demonstrates emergent flocking behavior from three simple rules:

  1. Separation: Avoid crowding neighbors

  2. Alignment: Steer toward average heading of neighbors

  3. Cohesion: Steer toward average position of neighbors

The CUDA implementation allows for real-time simulation of thousands of boids by parallelizing the computation across GPU cores.

## Usage
Program should be run using the Release version

Running the simulation:: boids_cuda_3d <computation_method> <number_of_boids>
<computation_method> - gpu or cpu
<number_of_boids> - number of boids

## Visualization
The simulation includes an OpenGL-based visualizer that shows:
- Individual boids cones
- Bounding box
- GUI in top left corner allowing for changing the values of certain parameters

You can use arrows to rotate the camera (right and left) or to get closer or further from the scene (up and down)

## References
- Reynolds, C. W. (1987). "Flocks, Herds, and Schools: A Distributed Behavioral Model". SIGGRAPH '87.
- NVIDIA CUDA Documentation: https://docs.nvidia.com/cuda/
- Boids algorithm overview: https://vergenet.net/~conrad/boids/pseudocode.html
