# author: g1n0st
# Jacobi Iterative Method for Mass-Spring System
from implicit_utility import *

# D = diag(A), E = A - D
# Ax = b
# Dx = -Ex + b
# x(i+1) = -D^-1Ex(i) + D-1b
@ti.kernel
def jacobi_iteration():
    for i in range(num_particles[None]):
        r = b[i]
        for j in range(num_particles[None]):
            if i != j:
                r -= A[i, j] @ v[j]

        new_v[i] = A[i, i].inverse() @ r

    for i in range(num_particles[None]):
        v[i] = new_v[i]

def substep(beta = 1.0, iter_times = 10):
    update_mass_matrix()
    update_jacobi_matrix()
    update_A_matrix(beta)
    update_F_vector()
    update_b_vector()

    for step in range(iter_times):
        jacobi_iteration()
    update_step()

init_mass_spring_system()

while True:
    process_input()

    if not paused[None]:
        for step in range(10):
            substep(1.0, 10)

    process_output()
