# author: g1n0st
# Runge-Kutta Method for Mass-Spring System
import sys
sys.path.append('../')
from mass_spring_framework import *

v2 = ti.Vector(2, dt = ti.f32, shape = max_particles)
v3 = ti.Vector(2, dt = ti.f32, shape = max_particles)
v4 = ti.Vector(2, dt = ti.f32, shape = max_particles)

a1 = ti.Vector(2, dt = ti.f32, shape = max_particles)
a2 = ti.Vector(2, dt = ti.f32, shape = max_particles)
a3 = ti.Vector(2, dt = ti.f32, shape = max_particles)
a4 = ti.Vector(2, dt = ti.f32, shape = max_particles)
    
@ti.kernel
def substep():
    # compute force and new velocity
    n = num_particles[None]

    for i in range(n):
        v[i] *= ti.exp(-dt * damping[None]) # damping

    #1
    for i in range(n):
        a1[i] = gravity * particle_mass
        for j in range(n):
            if rest_length[i, j] != 0:
                x_ij = x[i] - x[j]
                a1[i] += -stiffness[None] * (x_ij.norm() - rest_length[i, j]) * x_ij.normalized()

    #2
    for i in range(n):
        v2[i] = v[i] + (dt / 2) * a1[i]
        
    for i in range(n):
        a2[i] = gravity * particle_mass
        for j in range(n):
            if rest_length[i, j] != 0:
                x_ij = x[i] - x[j] + (dt / 2) * (v[i] - v[j])
                a2[i] += -stiffness[None] * (x_ij.norm() - rest_length[i, j]) * x_ij.normalized()

    #3
    for i in range(n):
        v3[i] = v[i] + (dt / 2) * a2[i]
        
    for i in range(n):
        a3[i] = gravity * particle_mass
        for j in range(n):
            if rest_length[i, j] != 0:
                x_ij = x[i] - x[j] + (dt / 2) * (v2[i] - v2[j])
                a3[i] += -stiffness[None] * (x_ij.norm() - rest_length[i, j]) * x_ij.normalized()

    #4
    for i in range(n):
        v4[i] = v[i] + dt * a3[i]
        
    for i in range(n):
        a4[i] = gravity * particle_mass
        for j in range(n):
            if rest_length[i, j] != 0:
                x_ij = x[i] - x[j] + dt * (v3[i] - v3[j])
                a4[i] += -stiffness[None] * (x_ij.norm() - rest_length[i, j]) * x_ij.normalized()

    collide_with_ground()

    for i in range(n):
        x[i] += (dt / 6) * ( v[i] + 2 * v2[i] + 2 * v3[i] + v4[i])
        v[i] += (dt / 6) * (a1[i] + 2 * a2[i] + 2 * a3[i] + a4[i]) / particle_mass

init_mass_spring_system()
total_time = 0.0
for frame in range(660):
    process_input(frame)

    start_time = time.time()
    if not paused[None]:
        for step in range(10):
            substep()
    end_time = time.time()
    print(end_time - start_time)
    total_time += end_time - start_time

    process_output(frame)

print('aver.')
print(total_time / 660 * 1000)
