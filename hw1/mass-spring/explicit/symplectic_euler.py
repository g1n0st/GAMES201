# author: g1n0st
# Symplectic Euler Method for Mass-Spring System
import sys
sys.path.append('../')
from mass_spring_framework import *

@ti.kernel
def substep():
    # compute force and new velocity
    n = num_particles[None]
    for i in range(n):
        v[i] *= ti.exp(-dt * damping[None]) # damping
        total_force = ti.Vector(gravity) * particle_mass
        for j in range(n):
            if rest_length[i, j] != 0:
                x_ij = x[i] - x[j]
                total_force += -stiffness[None] * (x_ij.norm() - rest_length[i, j]) * x_ij.normalized()
        v[i] += dt * total_force / particle_mass

    # collide with ground
    for i in range(n):
        if x[i].y < bottom_y:
            x[i].y = bottom_y
            v[i].y = 0

    # compute new position
    for i in range(n):
        x[i] += v[i] * dt

init_mass_spring_system()

while True:
    process_input()

    if not paused[None]:
        for step in range(10):
            substep()

    process_output()
