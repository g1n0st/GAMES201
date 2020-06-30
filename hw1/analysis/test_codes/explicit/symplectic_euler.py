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

    collide_with_ground()
    update_position()

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
