import taichi as ti
import numpy as np

ti.init(arch = ti.gpu)

MAX_GRID_NP = 96
MAX_NUM_NEIGHBORS = 96
MAX_NUM_PARTICLES = 5200

dim = 2 # dimension of the simulation
g = ti.Vector([0, -9.8]) # gravity
alpha0 = 0.30 # viscosity
rho_0 = 1000.0 # initial density of water
CFL_v = 0.25 # CFL coefficient for velocity
CFL_a = 0.05 # CFL coefficient for acceleration

dx = 0.1 # particle radius
dh = dx * 1.3 # smooth length
dt = 1e-4 # adaptive time step size according to CFL condition
m = dx ** dim * rho_0 # particle mass

# equation of state parameters
gamma = 7.0
c_0 = 200.0

res = (400, 400)
s2w = 35

cell_size = 2.0 * dh
grid_size = (np.ceil(np.array(res) / s2w / cell_size).astype(int) + 0xf) // 0x10 * 0x10 # scaling to 16x

# summing up the rho for all particles to compute the average rho
# sum_rho_err = ti.field(dtype = float, shape = ())
# sum_drho = ti.field(dtype = float, shape = ())

num_particles = ti.field(dtype = int, shape = ()) # total number of particles

# particles properties
x = ti.Vector.field(dim, dtype = float) # position
v = ti.Vector.field(dim, dtype = float) # velocity
# new_x = ti.Vector.field(dim, dtype = float) # prediction values for P-C scheme
# new_v = ti.Vector.field(dim, dtype = float) # prediction values for P-C scheme

P = ti.field(dtype = float) # pressure
rho = ti.field(dtype = float) # density
# new_rho = ti.field(dtype = float) # prediction values for P-C scheme

# alpha = ti.field(dtype = float)
# stiff = ti.field(dtype = float)
material = ti.field(dtype = float) # water particle / boundary particle

dv = ti.Vector.field(dim, dtype = float) # Dv/Dt
drho = ti.field(dtype = float) # Drho/Dt

grid_np = ti.field(dtype = int) # total number of particles in each grid
grid_p = ti.field(dtype = int) # particles ids of each grid
num_neighbors = ti.field(dtype = int) # total number of neighbors of each particle
neighbors = ti.field(dtype = int) # neighbors ids of each particle (not exceed maximum number of neighbors)

ti.root.dense(ti.i, MAX_NUM_PARTICLES).place(
        x, v, 
        # new_x, new_v,
        P, rho, 
        # new_rho,
        # alpha, stiff, 
        material,
        dv, drho)

# sparse structures
block_size = 16 # fix size align to 16

block0 = ti.root.pointer(ti.ij, grid_size // block_size)
block1 = block0.dense(ti.ij, block_size)
block1.place(grid_np)
block1.dense(ti.k, MAX_GRID_NP).place(grid_p)

block2 = ti.root.dense(ti.i, MAX_NUM_PARTICLES)
block2.place(num_neighbors)
block3 = block2.pointer(ti.j, MAX_NUM_NEIGHBORS // block_size)
block3.dense(ti.j, block_size).place(neighbors)

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
    for p in range(num_particles[None]):
        num_nb = 0
        if is_fluid(p) == 1 or is_fluid(p) == 0:
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
    res = 0.0
    if q < 1.0:
        res = (k / h) * (-3 * q + 2.25 * q ** 2)
    elif q < 2.0:
        res = -0.75 * (k / h) * (2 - q) ** 2
    return res

@ti.func
def delta_rho(p, q, r, norm_r):
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
def collision(p, n, d):
    # collision factor, assume roughly (1-c_f)*velocity loss after collision
    c_f = 0.3
    x[p] += n * d
    x[p] -= (1.0 + c_f) * v[p].dot(n) * n
    x[p] -= (1.0 + c_f) * new_v[p].dot(n) * n

gap = 6.0
@ti.kernel
def boundary_condition():
    for p in range(num_particles[None]):
        if is_fluid(p) == 1:
            for i in ti.static(range(dim)):
                normal = ti.Vector.zero(float, dim)
                normal[i] = 1.0
                if x[p][i] < gap:          v[p][i] = 0 # collision(p, normal, gap - x[p][i])
                normal[i] = -1.0
                if x[p][i] > res[i] - gap: v[p][i] = 0 # collision(p, normal, x[p][i] - res[i] + gap)

@ti.kernel
def wc_compute():
    for p in range(num_particles[None]):
        d_v = ti.Vector.zero(float, dim)
        d_rho = 0.0
        '''
        for j in range(num_neighbors[p]):
            q = neighbors[p, j]

            # compute distance and it's norm
            r = x[p] - x[q]
            norm = ti.max(r.norm(), 1e-5)

            # compute density change
            d_rho += delta_rho(p, q, r, norm)

            if is_fluid(p) == 1:
                # compute pressure force contribution
                d_v += pressure_force(p, q, r, norm)
        '''
        
        # add body force
        if is_fluid(p) == 1:
            d_v += g
        
        dv[p], drho[p] = d_v, d_rho

@ti.kernel
def wc_update():
    # forward euler
    for p in range(num_particles[None]):
        v[p] += dt * dv[p]
        x[p] += dt * v[p]
        rho[p] += dt * drho[p]
        P[p] = pressure(rho[p])

def substep():
    block1.deactivate_all()
    block0.deactivate_all()
    block2.deactivate_all()

    allocate_particles()
    search_neighbors()
    wc_compute()
    wc_update()
    boundary_condition()

@ti.kernel
def initialize():
    num_particles[None] = 50 * 50
    for i in range(50):
        for j in range(50):
            p = i * 50 + j
            x[p] = [i * dx + 2, j * dx + 2]
            v[p] = [0, -1]
            rho[p] = rho_0
            P[p] = pressure(rho_0)
            material[p] = 1

print(grid_size)
initialize()
gui = ti.GUI("2D Dam", res = res, background_color = 0xFFFFFF)
while True:
    for s in range(50):
        substep()
    
    gui.circles(x.to_numpy() / res * s2w, radius = 1.5, color = 0x068587)
    gui.show()

    