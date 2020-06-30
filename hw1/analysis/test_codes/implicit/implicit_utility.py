# author: g1n0st
# Utility for Implicit Method, including solve A and b (Ax = b)
import sys
sys.path.append('../')
from mass_spring_framework import *

# mass matrix of diag(m1, m2, ..., mn)
M = ti.Matrix(2, 2, dt = ti.f32, shape = (max_particles, max_particles))
# jacobi matrix of partial_f / partial_x
J = ti.Matrix(2, 2, dt = ti.f32, shape = (max_particles, max_particles))
# A = [M - beta * dt^2 * J]
A = ti.Matrix(2, 2, dt = ti.f32, shape = (max_particles, max_particles))
# force vector
F = ti.Vector(2, dt = ti.f32, shape = max_particles)
# b = dt * F
b = ti.Vector(2, dt = ti.f32, shape = max_particles)

# iteration temp variables
new_v = ti.Vector(2, dt = ti.f32, shape = max_particles)

@ti.kernel
def update_mass_matrix():
    m = ti.Matrix([
        [particle_mass, 0],
        [0, particle_mass]
    ])
    for i in range(num_particles[None]):
        M[i, i] = m

@ti.kernel
def update_jacobi_matrix():
    I = ti.Matrix([
        [1.0, 0.0],
        [0.0, 1.0]
        ])
    for i, d in J:
        J[i, d] *= 0.0
        for j in range(num_particles[None]):
            l_ij = rest_length[i, j]
            if (l_ij != 0) and (d == i or d == j):
                x_ij = x[i] - x[j]
                norm = x_ij.norm()
                unit_x_ij = x_ij / norm
                mat = unit_x_ij.outer_product(unit_x_ij)
                if d == i:
                    J[i, d] += -stiffness[None] * (I - l_ij / norm * (I - mat))
                else:
                    J[i, d] += stiffness[None] * (I - l_ij / norm * (I - mat))

@ti.kernel
def update_A_matrix(beta: ti.f32):
    for i, j in A:
        A[i, j] = M[i, j] - beta * dt**2 * J[i, j]

@ti.kernel
def update_F_vector():
    for i in range(num_particles[None]):
        F[i] = particle_mass * gravity
        
    for i, j in rest_length:
        if rest_length[i, j] != 0:
            x_ij = x[i] - x[j]
            F[i] += -stiffness[None] * (x_ij.norm() - rest_length[i, j]) * x_ij.normalized()
        else:
            pass

@ti.kernel
def update_b_vector():
    for i in range(num_particles[None]):
        v_star = v[i] * ti.exp(-dt * damping[None])
        b[i] = M[i, i] @ v_star + dt * F[i]

@ti.kernel
def update_step():
    collide_with_ground()
    update_position()
