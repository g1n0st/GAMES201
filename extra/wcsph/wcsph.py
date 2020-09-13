import taichi as ti
import numpy as np

ti.init(arch = ti.gpu)

MAX_GRID_NP = 96
MAX_NUM_NEIGHBORS = 128
MAX_NUM_PARTICLES = 10000

dim = 2 # dimension of the simulation
g = ti.Vector([0.0, -9.81]) # gravity
alpha0 = 0.08 # viscosity
rho_0 = 1000.0 # initial density of water
CFL_v = 0.25 # CFL coefficient for velocity
CFL_a = 0.05 # CFL coefficient for acceleration

dx = 0.1 # particle radius
dh = dx * 1.3 # smooth length
dt = 1e-4 # adaptive time step size according to CFL condition
m = dx ** dim * rho_0 # particle mass

# equation of state parameters
gamma = 7.0
c_s = 88.5 # speed of sound in the fluid

wd = (15.0, 15.0)
w2s = 1.0 / 15.0

cell_size = 2.0 * dh
grid_size = (np.ceil(np.array(wd) / cell_size).astype(int) + 0xf) // 0x10 * 0x10 # scaling to 16x

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
        idx = (x[p] / cell_size).cast(int) # get cell index of particle's position
        offset = ti.atomic_add(grid_np[idx], 1)
        grid_p[idx, offset] = p

@ti.kernel
def search_neighbors():
    for p in range(num_particles[None]):
        num_neighbors[p] = 0
        idx = (x[p] / cell_size).cast(int)
        for offset in ti.static(ti.grouped(ti.ndrange(*((-1, 2), ) * dim))): # [-1, 0, 1]
            pos = idx + offset
            if in_grid(pos) == 1:
                for k in range(grid_np[pos]):
                    if num_neighbors[p] >= MAX_NUM_NEIGHBORS: break
                    q = grid_p[pos, k]
                    if p != q and (x[p] - x[q]).norm() < cell_size:
                        nb = ti.atomic_add(num_neighbors[p], 1)
                        neighbors[p, nb] = q

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
    return rho_0 * c_s ** 2 / gamma * ((rho / rho_0) ** gamma - 1) # equation of state (EOS)

@ti.func
def pressure_force(p, q, r, norm_r):
    # compute the pressure force contribution, symmetric formula
    return -m * (P[p] / rho[p] ** 2 + P[q] / rho[q] ** 2) * dW(norm_r, dh) * r / norm_r

@ti.func
def viscosity_force(p, q, r, norm_r):
    res = ti.Vector.zero(float, dim)
    v_xy = (v[p] - v[q]).dot(r)
    if v_xy < 0.0:
        # artifical viscosity
        vmu = -2.0 * alpha0 * dx * c_s / (rho[p] + rho[q])
        res = -m * vmu * v_xy / (norm_r ** 2 + 0.01 * dx ** 2) * dW(norm_r, dh) * r / norm_r
    return res

gap = 0.3
@ti.kernel
def boundary_condition():
    for p in range(num_particles[None]):
        if is_fluid(p) == 1:
            for i in ti.static(range(dim)):
                # normal = ti.Vector.zero(float, dim)
                # normal[i] = 1.0
                if x[p][i] < gap and v[p][i] < 0:         v[p][i] = 0 # collision(p, normal, gap - x[p][i])
                # normal[i] = -1.0
                if x[p][i] > wd[i] - gap and v[p][i] > 0: v[p][i] = 0 # collision(p, normal, x[p][i] - wd[i] + gap)

@ti.kernel
def wc_compute():
    for p in range(num_particles[None]):
        d_v = ti.Vector.zero(float, dim)
        d_rho = 0.0

        for j in range(num_neighbors[p]):
            q = neighbors[p, j]

            # compute distance and it's norm
            r = x[p] - x[q]
            norm = ti.max(r.norm(), 1e-5)

            # compute density change
            d_rho += delta_rho(p, q, r, norm)

            if is_fluid(p) == 1:
                # compute Viscosity force contribution
                d_v += viscosity_force(p, q, r, norm)

                # compute pressure force contribution
                d_v += pressure_force(p, q, r, norm)
        
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
        P[p] = ti.max(pressure(rho[p]), 0)

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
    block_size = 50
    num_particles[None] = block_size ** 2
    for i in range(block_size):
        for j in range(block_size):
            p = i * block_size + j
            x[p] = [i * dx + 4.0, j * dx + 1.0] # collapsing column of water, height H = 4m
            v[p] = [0, -1.0]
            rho[p] = rho_0
            P[p] = pressure(rho_0)
            material[p] = 1
    
    wall_size = 5
    # down_wall
    off = num_particles[None]
    for i in range(150):
        for k in range(wall_size):
            p = off + i * wall_size + k
            x[p] = [i * dx + 0.05, k * dx + 0.05]
            v[p] = [0, 0]
            rho[p] = rho_0
            P[p] = pressure(rho_0)
            material[p] = 0
    num_particles[None] += 150 * wall_size
    off = num_particles[None]

    # left wall
    for i in range(140):
        for k in range(wall_size):
            p = off + i * wall_size + k
            x[p] = [k * dx + 0.05, i * dx + 0.55]
            v[p] = [0, 0]
            rho[p] = rho_0
            P[p] = pressure(rho_0)
            material[p] = 0
    num_particles[None] += 140 * wall_size
    off = num_particles[None]

    # right wall
    for i in range(140):
        for k in range(wall_size):
            p = off + i * wall_size + k
            x[p] = [15 - 0.05 - k * dx, i * dx + 0.55]
            v[p] = [0, 0]
            rho[p] = rho_0
            P[p] = pressure(rho_0)
            material[p] = 0
    num_particles[None] += 140 * wall_size
    off = num_particles[None]

    # top wall
    for i in range(150):
        for k in range(wall_size):
            p = off + i * wall_size + k
            x[p] = [i * dx + 0.05, 15.0 - 0.05 - k * dx]
            v[p] = [0, 0]
            rho[p] = rho_0
            P[p] = pressure(rho_0)
            material[p] = 0
    num_particles[None] += 150 * wall_size
    off = num_particles[None]

initialize()
gui = ti.GUI("wcsph", res = 512, background_color = 0xFFFFFF)

frame = 0
while gui.running:
    for s in range(20):
        substep()

    colors = np.array([0x855E42, 0x068587], dtype = np.uint32)
    gui.circles(x.to_numpy() * w2s, radius = 2, color = colors[material.to_numpy()])

    '''
    # target particle
    target = 514
    gui.circle([x[target][0] * w2s, x[target][1] * w2s], radius = 2, color = 0xFF0000)
    for i in range(num_neighbors[target]):
        gui.circle([x[neighbors[target, i]][0] * w2s, x[neighbors[target, i]][1] * w2s], radius = 2, color = 0xFFFF00)
    '''

    if frame % 10 == 0: gui.show(f'{frame // 10:06d}.png')
    else: gui.show()
    frame += 1