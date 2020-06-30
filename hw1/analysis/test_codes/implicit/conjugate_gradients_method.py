# author: g1n0st
# Conjugate Gradients Method for Mass-Spring System
from implicit_utility import *

d = ti.Vector(2, dt = ti.f32, shape = max_particles)
r = ti.Vector(2, dt = ti.f32, shape = max_particles)
a = ti.var(dt = ti.f32, shape = max_particles)

# d_0 = r_0 = b - Ax_0
@ti.kernel
def conjugate_gradients_init():
    for i in range(num_particles[None]):
        Ax_i = ti.Vector([0.0, 0.0])
        for j in range(num_particles[None]):
            Ax_i += A[i, j] @ v[j]
        d[i] = b[i] - Ax_i
        r[i] = b[i] - Ax_i

# a_i = rT_i * r_i / (dT_i * A * d_i)
# x_i+1 = x_i + a_i * d_i
# r_i+1 = r_i - a_i * A * d_i
# b_i+1 = rT_i+1 * r_i+1 / (rT_i * r_i)
# d_i+1 = r_i+1 + b_i+1 * d_i
@ti.kernel
def conjugate_gradients():
    for i in range(num_particles[None]):
        a[i] = r[i].dot(r[i])
        dTA_i = ti.Vector([0.0, 0.0])
        for j in range(num_particles[None]):
            dTA_i[0] += d[i][0] * A[j, i][0, 0] + d[i][1] * A[j, i][1, 0]
            dTA_i[1] += d[i][1] * A[j, i][0, 1] + d[i][1] * A[j, i][1, 1]
            
        a[i] /= dTA_i.dot(d[i])

    for i in range(num_particles[None]):
        v[i] += a[i] * d[i]

    for i in range(num_particles[None]):
        Ad_i = ti.Vector([0.0, 0.0])
        for j in range(num_particles[None]):
            Ad_i += A[i, j] @ d[j]

        rr = r[i].dot(r[i])
        r[i] -= a[i] * Ad_i
        beta = r[i].dot(r[i]) / rr
        d[i] = r[i] + beta * d[i]

def substep(beta = 1.0, iter_times = 10):
    update_mass_matrix()
    update_jacobi_matrix()
    update_A_matrix(beta)
    update_F_vector()
    update_b_vector()

    conjugate_gradients_init()
    for step in range(iter_times):
        conjugate_gradients()
    update_step()

init_mass_spring_system()

while True:
    process_input()

    if not paused[None]:
        for step in range(10):
            substep(1.0, 1)

    process_output()
