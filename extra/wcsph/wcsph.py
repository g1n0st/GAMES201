import taichi as ti
import numpy as np

ti.init(arch = ti.gpu)

MAX_GRID_NP = 32
MAX_NUM_NEIGHBORS = 96
MAX_NUM_PARTICLES = 6400

dim = 2 # dimension of the simulation
g = ti.Vector([0.0, -9.81]) # gravity
alpha0 = 0.40 # viscosity
rho_0 = 1000.0 # initial density of water
CFL_v = 0.25 # CFL coefficient for velocity
CFL_a = 0.05 # CFL coefficient for acceleration

dx = 5e-4 # particle radius
dh = dx * 3.0 # smooth length
dt = 3e-5 # adaptive time step size according to CFL condition
m = dx ** dim * rho_0 # particle mass

# equation of state parameters
gamma = 7.0
c_s = 9.0 # speed of sound in the fluid

# surface tension parameters
c = 1.9e8 # a user-tuned coefficient that controls the strength of the interaction force
k = (8 / 3) ** (1 / 3) # the required enlargement ratio k for the original radius of the neighborhood h
kh = dh * k

wd = (15.0 * 5e-3, 15.0 * 5e-3)
w2s = 1.0 / (15.0 * 5e-3)

grid_size = (np.ceil(np.array(wd) / dh).astype(int) + 0xf) // 0x10 * 0x10 # scaling to 16x

num_particles = ti.field(dtype = int, shape = ()) # total number of particles

# particles properties
x = ti.Vector.field(dim, dtype = float) # position
v = ti.Vector.field(dim, dtype = float) # velocity
P = ti.field(dtype = float) # pressure
rho = ti.field(dtype = float) # density
material = ti.field(dtype = int) # water particle / boundary particle

dv = ti.Vector.field(dim, dtype = float) # Dv/Dt
drho = ti.field(dtype = float) # Drho/Dt

grid_np = ti.field(dtype = int) # total number of particles in each grid
grid_p = ti.field(dtype = int) # particles ids of each grid
num_neighbors = ti.field(dtype = int) # total number of neighbors of each particle
neighbors = ti.field(dtype = int) # neighbors ids of each particle (not exceed maximum number of neighbors)

ti.root.dense(ti.i, MAX_NUM_PARTICLES).place(x, v, P, rho, material, dv, drho)

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
def allocate_particles():
    for p in range(num_particles[None]):
        idx = (x[p] / kh).cast(int) # get cell index of particle's position
        offset = ti.atomic_add(grid_np[idx], 1)
        grid_p[idx, offset] = p

@ti.kernel
def search_neighbors():
    for p in range(num_particles[None]):
        num_neighbors[p] = 0
        idx = (x[p] / kh).cast(int)
        for offset in ti.static(ti.grouped(ti.ndrange(*((-1, 2), ) * dim))): # [-1, 0, 1]
            pos = idx + offset
            if in_grid(pos) == 1:
                for k in range(grid_np[pos]):
                    if num_neighbors[p] >= MAX_NUM_NEIGHBORS: break
                    q = grid_p[pos, k]
                    if p != q and (x[p] - x[q]).norm() < kh:
                        nb = ti.atomic_add(num_neighbors[p], 1)
                        neighbors[p, nb] = q

sigma = [0.0, 4 / (3 * dh), 40 / (7 * np.pi * dh ** 2), 8 / (np.pi * dh ** 3)]
s_d = sigma[dim]
# cubic spline smoothing kernel
@ti.func
def W(r, h):
    res = 0.0
    q = r / h
    if q <= 0.5: res = 6 * (q ** 3 - q ** 2) + 1
    elif q <= 1.0: res = 2 * (1 - q) ** 3
    return s_d * res
# derivative of cubcic spline smoothing kernel
@ti.func
def dW(r, h):
    res = 0.0
    q = r / h
    if q <= 0.5: res = 6 * (3 * q ** 2 - 2 * q)
    elif q <= 1.0: res = -6 * (1 - q) ** 2
    return (s_d / h) * res

@ti.func
def delta_rho(p, q, r, norm_r):
    # density delta, i.e. divergence
    return (m / rho[q]) * dW(norm_r, dh) * (v[p] - v[q]).dot(r / norm_r) # from J.J. Monaghan, Smoothed particle hydrodynamics

@ti.func
def pressure(rho):
    return rho_0 * c_s ** 2 / gamma * ((rho / rho_0) ** gamma - 1) # equation of state (EOS)

@ti.func
def pressure_force_(p, q, r, norm_r):
    # compute the pressure force contribution, symmetric formula
    return -m * (P[p] / rho[p] ** 2 + P[q] / rho[q] ** 2) * dW(norm_r, dh) * r / norm_r

@ti.func
def pressure_force(p, q, r, norm_r):
    p_ab = (rho[q] * P[p] + rho[p] * P[q]) / (rho[p] + rho[q]) # density-weighted inter-particle averaged pressure
    return - 1 / m * ((m / rho[p]) ** 2 + (m / rho[q]) ** 2) * p_ab * dW(norm_r, dh) * r / norm_r # according to Hu and Adams
@ti.func
def viscosity_force(p, q, r, norm_r):
    res = ti.Vector.zero(float, dim)
    v_xy = (v[p] - v[q]).dot(r)
    if v_xy < 0.0:
        # artifical viscosity
        vmu = -2.0 * alpha0 * dx * c_s / (rho[p] + rho[q])
        res = -m * vmu * v_xy / (norm_r ** 2 + 0.01 * dx ** 2) * dW(norm_r, dh) * r / norm_r
    return res

@ti.func
def pairwise_force(p, q, r, norm_r):
    fr = 0.0
    if norm_r <= kh: fr = ti.cos(3 * np.pi * norm_r / (2 * kh))
    return c * m ** 2 * fr * r / norm_r

gap = 0.3 * 5e-3
@ti.kernel
def boundary_condition():
    for p in range(num_particles[None]):
        if is_fluid(p) == 1:
            for i in ti.static(range(dim)):
                if x[p][i] < gap and v[p][i] < 0:         v[p][i] = 0
                if x[p][i] > wd[i] - gap and v[p][i] > 0: v[p][i] = 0

@ti.kernel
def update_vx():
    for p in range(num_particles[None]):
        v[p] += (dt / 2) * dv[p]
        x[p] += (dt / 2) * v[p]

@ti.kernel
def update_rhox():
    for p in range(num_particles[None]):
        drho[p] = 0
        for j in range(num_neighbors[p]):
            q = neighbors[p, j]
            r = x[p] - x[q]
            norm = ti.max(r.norm(), 1e-5) # compute distance and it's norm
            drho[p] += rho[p] * delta_rho(p, q, r, norm) # compute density change

    for p in range(num_particles[None]):
        rho[p] += dt * drho[p]
        P[p] = pressure(rho[p])
        x[p] += (dt / 2) * v[p]

eta = 0.2 # a user-defined coefficient to determine the degree of anisotropy
eps_min = 0.5 # user-defined minimum value of the eigenvalue
@ti.kernel
def update_vdv():
    for p in range(num_particles[None]):
        dv[p] = ti.Vector.zero(float, dim)
        # anisotropic filtering parameters
        C = ti.Matrix.zero(float, dim, dim) # anisotropic covariance
        w = 0.0
        pf = ti.Vector.zero(float, dim) # total pairwise force
        for j in range(num_neighbors[p]):
            q = neighbors[p, j]
            r = x[p] - x[q]
            norm = ti.max(r.norm(), 1e-5) # compute distance and it's norm
            if is_fluid(p) == 1:
                dv[p] += viscosity_force(p, q, r, norm) # compute Viscosity force contribution
                dv[p] += pressure_force(p, q, r, norm) # compute pressure force contribution
                w += W(norm, dh) 
                C += r.outer_product(r) * w
                pf += pairwise_force(p, q, r, norm) # compute surface tension contribution
        
        # add body force and filtered pairwise force
        if is_fluid(p) == 1:
            dv[p] += g
        
            C /= w
            U, sig, V = ti.svd(C)
            for i in ti.static(range(dim)):
                sig[i, i] = ti.max(sig[i, i], eps_min) 

            G = U @ sig.inverse() @ V.transpose()
            T = (1 - eta) * ti.Matrix.identity(float, dim) + eta * G / G.determinant()
            dv[p] += T @ pf
    
    for p in range(num_particles[None]):
        v[p] += (dt / 2) * dv[p]

def update_neighbors():
    block1.deactivate_all()
    block0.deactivate_all()
    block2.deactivate_all()
    grid_np.fill(0)
    num_neighbors.fill(0)
    allocate_particles()
    search_neighbors()

def substep():
    update_vx() # velocity-Verlet scheme
    update_neighbors()
    update_rhox()
    update_neighbors()
    update_vdv()
    boundary_condition()

@ti.func
def add_particle(position, velocity, mat):
    p = num_particles[None]
    num_particles[None] += 1
    x[p] = position
    v[p] = velocity
    rho[p] = rho_0
    P[p] = pressure(rho_0)
    material[p] = mat

@ti.kernel
def initialize():
    block_size = 50
    for i in range(block_size):
        for j in range(block_size):
            add_particle([i * dx + 4.0 * 5e-3, j * dx + 7.0 * 5e-3], [0, 0], 1)
    
    wall_size = 5
    # down_wall
    for i in range(150):
        for k in range(wall_size):
            add_particle([i * dx + 0.05 * 5e-3, k * dx + 0.05 * 5e-3], [0, 0], 0)

    # left wall
    for i in range(140):
        for k in range(wall_size):
            add_particle([k * dx + 0.05 * 5e-3, i * dx + 0.55 * 5e-3], [0, 0], 0)

    # right wall
    for i in range(140):
        for k in range(wall_size):
            add_particle([(15 - 0.05) * 5e-3 - k * dx, i * dx + 0.55 * 5e-3], [0, 0], 0)

    # top wall
    for i in range(150):
        for k in range(wall_size):
            add_particle([i * dx + 0.05 * 5e-3, (15.0 - 0.05) * 5e-3 - k * dx], [0, 0], 0)

@ti.kernel
def gank():
    block_size = 8
    for i in range(block_size):
        for j in range(block_size):
            add_particle([i * dx + 4.0 * 5e-3, j * dx + 12.5 * 5e-3], [0, -350.0 * 5e-3], 0)

initialize()
gui = ti.GUI("wcsph", res = 512, background_color = 0xFFFFFF)

frame = 0
while gui.running:
    if frame == 1250: gank()
    for s in range(20):
        substep()

    colors = np.array([0x855E42, 0x068587], dtype = np.uint32)
    gui.circles(x.to_numpy() * w2s, radius = 2, color = colors[material.to_numpy()])

    if frame % 10 == 0: gui.show(f'{frame // 10:06d}.png')
    else: gui.show()
    frame += 1