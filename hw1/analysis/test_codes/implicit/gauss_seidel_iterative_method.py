# author: g1n0st
# Gauss-Seidel Iterative Method for Mass-Spring System
from implicit_utility import *

# temp data for gauss-seidel iterative method
inv_L = ti.Matrix(2, 2, dt = ti.f32, shape = (max_particles, max_particles))
invL_v = ti.Vector(2, dt = ti.f32, shape = (max_particles))

@ti.kernel
def gauss_seidel_init():
    O = ti.Matrix([
        [0.0, 0.0],
        [0.0, 0.0]
    ])
    I = ti.Matrix([
        [1.0, 0.0],
        [0.0, 1.0]
        ])

    for i, j in inv_L:
        inv_L[i, j] = O
    for i in range(num_particles[None]):
        inv_L[i, i] = I

    # solve L-1, inv of A's upper triangular matrix
    for i in range(num_particles[None]):
        inv_ii = A[i, i].inverse()
        for j in range(i):
            n = A[j, i] @ inv_ii
            inv_L[j, i] -= n

    for i in range(num_particles[None]):
        inv_ii = A[i, i].inverse()
        for j in range(i, num_particles[None]):
            inv_L[i, j] = inv_L[i, j] @ inv_ii

# D = diag(A), L = -tril(A), U = -triu(A)
# (D - L)x(i+1) = Ux(i) + b
# x(i+1) = (D - L)^-1 * (Ux(i) + b)
@ti.kernel
def gauss_seidel_iteration():
    # solve Ux(i) + b
    for i in range(num_particles[None]):
        new_v[i] = b[i]
        for j in range(i):
            new_v[i] -= A[i, j] @ v[j]

    # solve (D - L)^-1 * (Ux(i) + b)
    for i in range(num_particles[None]):
        invL_v[i] = ti.Vector([0.0, 0.0])
        for j in range(i, num_particles[None]):
            invL_v[i] += inv_L[i, j] @ new_v[j]

    # update v
    for i in range(num_particles[None]):
        v[i] = invL_v[i]

def substep(beta = 1.0, iter_times = 10):
    update_mass_matrix()
    update_jacobi_matrix()
    update_A_matrix(beta)
    update_F_vector()
    update_b_vector()

    gauss_seidel_init()
    for step in range(iter_times):
        gauss_seidel_iteration()
    update_step()

init_mass_spring_system()
total_time = 0.0
for frame in range(660):
    process_input(frame)

    start_time = time.time()
    if not paused[None]:
        for step in range(10):
            substep(1.0, 10)
    end_time = time.time()
    print(end_time - start_time)
    total_time += end_time - start_time

    process_output(frame)

print('aver.')
print(total_time / 660 * 1000)
