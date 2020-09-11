import taichi as ti
import numpy as np

ti.init(arch = ti.gpu)

DYNAMIC_ALLOCATE = True
MAX_GRID_NP = 100
MAX_NUM_NEIGHBORS = 100
MAX_NUM_PARTICLES = 2 ** 20

dim = 2 # dimension of the simulation
g = -9.80 # gravity
alpha0 = 0.30 # viscosity
rho0 = 1000.0 # initial density of water
CFL_v = 0.25 # CFL coefficient for velocity
CFL_a = 0.05 # CFL coefficient for acceleration

dx = 0.1 # particle radius
dh = dx * 1.3 # smooth length
dt = ti.field(dtype = float, shape = ()) # adaptive time step size according to CFL condition
m = dx ** dim * rho0 # particle mass

# equation of state parameters
gamma = 7.0
c_0 = 200.0

res = (400, 400)
s2w = 50.0

cell_size = 2 * dh
grid_size = np.ceil(np.array(res) / s2w / cell_size).astype(int)

# summing up the rho for all particles to compute the average rho
sum_rho_err = ti.field(dtype = float, shape = ())
sum_drho = ti.field(dtype = float, shape = ())

num_particles = ti.field(dtype = int, shape = ()) # total number of particles

# particles properties
x = ti.Vector.field(dim, dtype = float) # position
v = ti.Vector.field(dim, dtype = float) # velocity
new_x = ti.Vector.field(dim, dtype = float) # prediction values for P-C scheme
new_v = ti.Vector.field(dim, dtype = float) # prediction values for P-C scheme

P = ti.field(dtype = float) # pressure
rho = ti.field(dtype = float) # density
new_rho = ti.field(dtype = float) # prediction values for P-C scheme

alpha = ti.field(dtype = float)
stiff = ti.field(dtype = float)
material = ti.field(dtype = float) # water particle / boundary particle

dv = ti.Vector.field(dim, dtype = float) # Dv/Dt
drho = ti.field(dtype = float) # Drho/Dt

grid_np = ti.field(dtype = int) # total number of particles in each grid
grid_p = ti.field(dtype = int) # particles ids of each grid
num_neighbors = ti.field(dtype = int) # total number of neighbors of each particle
neighbors = ti.field(dtype = int) # neighbors ids of each particle (not exceed maximum number of neighbors)

if ti.static(DYNAMIC_ALLOCATE):
    ti.root.dynamic(ti.i, MAX_NUM_PARTICLES, 2 ** 18).place(
            x, v, new_x, new_v,
            P, rho, new_rho,
            alpha, stiff, material,
            dv, drho)
else:
    ti.root.dense(ti.i, 2 ** 18).place(
            x, v, new_x, new_v,
            P, rho, new_rho,
            alpha, stiff, material,
            dv, drho)

if ti.static(dim == 2):
    snode = ti.root.dense(ti.ij, grid_size)
    snode.place(grid_np)
    snode.dense(ti.k, MAX_GRID_NP).place(grid_p)
else:
    snode = ti.root.dense(ti.ijk, grid_size)
    snode.place(grid_np)
    snode.dense(ti.l, MAX_GRID_NP).place(grid_p)

nb_node = ti.root.dynamic(ti.i, MAX_NUM_PARTICLES)
nb_node.place(num_neighbors)
nb_node.dense(ti.j, MAX_NUM_NEIGHBORS).place(neighbors)

# initialize dt
dt.from_numpy(np.array(1.0 * dh / c_0, dtype = np.float32))

@ti.kernel
def allocate_particles():
    for p in range(num_particles[None]):
        idx = (x[p] / cell_size).cast(int) # get cell index of particle's position
        offset = grid_np[idx].atomic_add(1)
        grid_p[idx, offset] = p

@ti.func
def in_grid(c):
    res = 1
    for i in ti.static(range(dim)):
        res = ti.atomic_and(res, (0 <= c[i] < grid_size[i]))
    return res

@ti.func
def is_fluid(p):
    # check whether fluid particle or boundary particle
    return material[p]

@ti.kernel
def search_neighbors():
    for p in range(num_particle[None]):
        num_nb = 0
        if is_fluid(x[p]) == 1 or is_fluid(x[p]) == 0:
            idx = (x[p] / cell_size).cast(int)
            for offset in ti.static(ti.grouped(ti.ndrange(*((-1, 2), ) * dim))):
                if in_grid(idx + offset) == 1:
                    for j in range(grid_np[idx + offset]):
                        if num_nb >= MAX_NUM_NEIGHBORS: break
                        q = grid_p[idx + offset, j]
                        if p != q and (x[p] - x[q]).norm() < cell_size:
                            neighbors[p, num_nb] = q
                            num_nb.atomic_add(1)

        num_neighbors[p] = num_nb

@ti.func
def W(r, h):
    # value of cubic spline smoothing kernel
    k = 10.0 / (7.0 * np.pi * h ** dim)
    q = r / h
    res = 0.0
    if q <= 1.0: 
        res = k * (1 - 1.5 * q ** 2 + 0.75 * q ** 3)
    elif q < 2.0: 
        res = k * 0.25 * (2 - q) ** 3
    return res

@ti.func
def dW(r, h):
    # derivative of cubcic spline smoothing kernel
    k = 10.0 / (7.0 * np.pi * h ** dim)
    q = r / h
    if q < 1.0:
        res = (k / h) * (-3 * q + 2.25 * q ** 2)
    elif q < 2.0:
        res = -0.75 * (k / h) * (2 - q) ** 2
    return res

@ti.func
def d_rho(p, q, r, norm_r):
    # density delta, i.e. divergence
    return m * dW(norm_r, dh) * (v[p] - v[q]).dot(r / norm_r) # formula (6)

@ti.func
def pressure(rho):
    return rho_0 * c_0 ** 2 / gamma * ((rho / rho_0) ** gamma - 1) # equation of state (EOS)

@ti.func
def pressure_force(p, q, r, norm_r):
    # compute the pressure force contribution, symmetric formula
    return -m * (P[p] / rho[p] ** 2 + P[q] / rho[q] ** 2) * dW(norm_r, dh) * r / norm_r

@ti.func
def simulate_collisions(p, n, d):
    # collision factor, assume roughly (1-c_f)*velocity loss after collision
    c_f = 0.3
    x[p] += v * d
    x[p] -= (1.0 + c_f) * v[p].dot(n) * n
    x[p] -= (1.0 + c_f) * new_v[p].dot(n) * n

