import taichi as ti
import numpy as np
import time
ti.init(arch=ti.gpu)

quality = 1
n_s_particles, n_w_particles, n_grid = 9000 * quality ** 2, 9000 * quality ** 2, 128 * quality
dx, inv_dx = 1 / n_grid, float(n_grid)
dt = 1e-4 / quality
gravity = ti.Vector([0, -98])
d = 2

# sand particle properties
x_s = ti.Vector.field(2, dtype = float, shape = n_s_particles) # position
v_s = ti.Vector.field(2, dtype = float, shape = n_s_particles) # velocity
C_s = ti.Matrix.field(2, 2, dtype = float, shape = n_s_particles) # affine velocity matrix
F_s = ti.Matrix.field(2, 2, dtype = float, shape = n_s_particles) # deformation gradient
c_C = ti.field(dtype = float, shape = n_s_particles) # cohesion and saturation
c_C0 = ti.field(dtype = float, shape = n_s_particles) # initial cohesion (as maximum)
vc_s = ti.field(dtype = float, shape = n_s_particles) # tracks changes in the log of the volume gained during extension
alpha_s = ti.field(dtype = float, shape = n_s_particles) # yield surface size
q_s = ti.field(dtype = float, shape = n_s_particles) # harding state

# sand grid properties
grid_sv = ti.Vector.field(2, dtype = float, shape = (n_grid, n_grid)) # grid node momentum/velocity
grid_sm = ti.field(dtype = float, shape = (n_grid, n_grid)) # grid node mass
grid_sf = ti.Vector.field(2, dtype = float, shape = (n_grid, n_grid)) # forces in the sand

# water particle properties
x_w = ti.Vector.field(2, dtype = float, shape = n_w_particles) # position
v_w = ti.Vector.field(2, dtype = float, shape = n_w_particles) # velocity
C_w = ti.Matrix.field(2, 2, dtype = float, shape = n_w_particles) # affine velocity matrix
J_w = ti.field(dtype = float, shape = n_w_particles) # ratio of volume increase

# water grid properties
grid_wv = ti.Vector.field(2, dtype = float, shape = (n_grid, n_grid)) # grid node momentum/velocity
grid_wm = ti.field(dtype = float, shape = (n_grid, n_grid)) # grid node mass
grid_wf = ti.Vector.field(2, dtype = float, shape = (n_grid, n_grid)) #  forces in the water

# constant values
p_vol, s_rho, w_rho = (dx * 0.5) ** 2, 2200, 2
s_mass, w_mass = p_vol * s_rho, p_vol * w_rho

w_k, w_gamma = 50, 3 # bulk modulus of water and gamma is a term that more stiffy penalizes large deviations from incompressibility

n, k_hat = 0.4, 0.2 # sand porosity and permeability

E_s, nu_s = 3.537e5, 0.3 # sand's Young's modulus and Poisson's ratio
mu_s, lambda_s = E_s / (2 * (1 + nu_s)), E_s * nu_s / ((1 + nu_s) * (1 - 2 * nu_s)) # sand's Lame parameters

a, b, c0, sC = -3.0, 0, 1e-2, 0.15
# The scalar function h_s is chosen so that the multiplier function is twice continuously differentiable
@ti.func
def h_s(z):
    ret = 0.0
    if z < 0: ret = 1
    if z > 1: ret = 0
    ret = 1 - 10 * (z ** 3) + 15 * (z ** 4) - 6 * (z ** 5)
    return ret

# multiplier
sqrt2 = ti.sqrt(2)
@ti.func
def h(e):
    u = e.trace() / sqrt2
    v = ti.abs(ti.Vector([e[0, 0] - u / sqrt2, e[1, 1] - u / sqrt2]).norm())
    fe = c0 * (v ** 4) / (1 + v ** 3)

    ret = 0.0
    if u + fe < a + sC: ret = 1
    if u + fe > b + sC: ret = 0
    ret = h_s((u + fe - a - sC) / (b - a))
    return ret

h0, h1, h2, h3 = 35, 9, 0.2, 10
pi = 3.14159265358979
@ti.func
def project(e, cC, p):
    ehat = e - e.trace() / d * ti.Matrix.identity(float, 2)
    yp = ehat.determinant() + (d * lambda_s + 2 * mu_s) / (2 * mu_s) * e.trace() * alpha_s[p]
    new_e = ti.Matrix.zero(float, 2, 2)
    delta_q = 0.0
    if ehat.determinant() == 0 or e.trace() > 0:
        new_e = ti.Matrix.zero(float, 2, 2)
        delta_q = e.determinant()
    elif yp <= 0:
        new_e = e
        delta_q = 0
    else:
        new_e = e - yp * ehat / ehat.determinant()
        delta_q = yp
    q_s[p] += delta_q
    phi = h0 + (h1 * q_s[p] - h3) * ti.exp(-h2 * q_s[p])
    phi = ti.min(pi / 2, ti.max(0, phi)) # must to do ?        
    sin_phi = ti.sin(phi)
    alpha_s[p] = ti.sqrt(2 / 3) * (2 * sin_phi) / (3 - sin_phi)

    return new_e

@ti.kernel
def substep():
    # set zero initial state for both water/sand grid
    for i, j in grid_sm:
        grid_sv[i, j], grid_wv[i, j] = [0, 0], [0, 0]
        grid_sm[i, j], grid_wm[i, j] = 0, 0
        grid_sf[i, j], grid_wf[i, j] = [0, 0], [0, 0]

    # P2G (sand's part)
    for p in x_s:
        base = (x_s[p] * inv_dx - 0.5).cast(int)
        if base[0] < 0 or base[1] < 0 or base[0] >= n_grid - 2 or base[1] >= n_grid - 2:
            continue
        fx = x_s[p] * inv_dx - base.cast(float)
        # Quadratic kernels  [http://mpm.graphics   Eqn. 123, with x=fx, fx-1,fx-2]
        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]
        U, sig, V = ti.svd(F_s[p])
        inv_sig = sig.inverse()
        e = ti.Matrix([[ti.log(sig[0, 0]), 0], [0, ti.log(sig[1, 1])]])
        stress = U @ ((2 * mu_s * inv_sig) @ e + lambda_s * e.trace() * inv_sig) @ V.transpose() # formula (25)
        stress = (-p_vol * 4 * inv_dx * inv_dx) * stress @ F_s[p].transpose()
        stress *= h(e)
        # print(h(e))
        affine = s_mass * C_s[p]
        for i, j in ti.static(ti.ndrange(3, 3)):
            offset = ti.Vector([i, j])
            dpos = (offset.cast(float) - fx) * dx
            weight = w[i][0] * w[j][1]
            grid_sv[base + offset] += weight * (s_mass * v_s[p] + affine @ dpos)
            grid_sm[base + offset] += weight * s_mass
            grid_sf[base + offset] += weight * stress @ dpos

    '''
    # P2G (water's part):
    for p in x_w:
        base = (x_w[p] * inv_dx - 0.5).cast(int)
        fx = x_w[p] * inv_dx - base.cast(float)
        # Quadratic kernels  [http://mpm.graphics   Eqn. 123, with x=fx, fx-1,fx-2]
        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]
        stress = w_k * (1 - 1 / (J_w[p] ** w_gamma))
        stress = (-p_vol * 4 * inv_dx * inv_dx) * stress * J_w[p]
        # stress = -4 * 400 * p_vol * (J_w[p] - 1) / dx ** 2 (special case when gamma equals to 1)
        affine = w_mass * C_w[p]
        # affine = ti.Matrix([[stress, 0], [0, stress]]) + w_mass * C_w[p]
        for i, j in ti.static(ti.ndrange(3, 3)):
            offset = ti.Vector([i, j])
            dpos = (offset.cast(float) - fx) * dx
            weight = w[i][0] * w[j][1]
            grid_wv[base + offset] += weight * (w_mass * v_w[p] + affine @ dpos)
            grid_wm[base + offset] += weight * w_mass
            grid_wf[base + offset] += weight * stress * dpos
    '''

    # Update Grids Momenta
    for i, j in grid_sm:
        if grid_sm[i, j] > 0:
            grid_sv[i, j] = (1 / grid_sm[i, j]) * grid_sv[i, j] # Momentum to velocity
            grid_sv[i, j] += dt * (gravity + grid_sf[i, j] / grid_sm[i, j]) # Update acceleration
            # grid_sv[i, j] += dt * gravity
        '''
        if grid_wm[i, j] > 0:
            grid_wv[i, j] = (1 / grid_wm[i, j]) * grid_wv[i, j]
            grid_wv[i, j] += dt * (gravity + grid_wf[i, j] / grid_wm[i, j])
        if grid_sm[i, j] > 0 and grid_wm[i, j] > 0:
            cE = (n * n * w_rho * gravity[1]) / k_hat
            p_s = cE * (grid_wv[i, j] - grid_sv[i, j])
            grid_sv[i, j] += p_s / grid_sm[i, j]
            grid_wv[i, j] -= p_s / grid_wm[i, j]
        '''

        if grid_sm[i, j] > 0:
            if i < 3 and grid_sv[i, j][0] < 0:          grid_sv[i, j][0] = 0 # Boundary conditions
            if i > n_grid - 3 and grid_sv[i, j][0] > 0: grid_sv[i, j][0] = 0
            if j < 3 and grid_sv[i, j][1] < 0:          grid_sv[i, j][1] = 0
            if j > n_grid - 3 and grid_sv[i, j][1] > 0: grid_sv[i, j][1] = 0
        '''
        if grid_wm[i, j] > 0:
            if i < 3 and grid_wv[i, j][0] < 0:          grid_wv[i, j][0] = 0 # Boundary conditions
            if i > n_grid - 3 and grid_wv[i, j][0] > 0: grid_wv[i, j][0] = 0
            if j < 3 and grid_wv[i, j][1] < 0:          grid_wv[i, j][1] = 0
            if j > n_grid - 3 and grid_wv[i, j][1] > 0: grid_wv[i, j][1] = 0
        '''

    # G2P (sand's part)
    for p in x_s:
        base = (x_s[p] * inv_dx - 0.5).cast(int)
        if base[0] < 0 or base[1] < 0 or base[0] >= n_grid - 2 or base[1] >= n_grid - 2:
            continue
        fx = x_s[p] * inv_dx - base.cast(float)
        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1.0) ** 2, 0.5 * (fx - 0.5) ** 2]
        new_v = ti.Vector.zero(float, 2)
        new_C = ti.Matrix.zero(float, 2, 2)
        phi = 0
        for i, j in ti.static(ti.ndrange(3, 3)): # loop over 3x3 grid node neighborhood
            dpos = ti.Vector([i, j]).cast(float) - fx
            g_v = grid_sv[base + ti.Vector([i, j])]
            weight = w[i][0] * w[j][1]
            new_v += weight * g_v
            new_C += 4 * inv_dx * weight * g_v.outer_product(dpos)
            if grid_sm[i, j] > 0 and grid_wm[i, j] > 0:
                phi += weight # formula (24)

        F_s[p] = (ti.Matrix.identity(float, 2) + dt * new_C) @ F_s[p]
        v_s[p], C_s[p] = new_v, new_C
        x_s[p] += dt * v_s[p]

        c_C[p] = c_C0[p] * (1 - phi)
        U, sig, V = ti.svd(F_s[p])
        e = ti.Matrix([[ti.log(sig[0, 0]), 0], [0, ti.log(sig[1, 1])]])
        new_e = project(e + vc_s[p] / d * ti.Matrix.identity(float, 2), c_C[p], p)
        # new_e = project(e, c_C[p], p)
        new_F = U @ ti.Matrix([[ti.exp(new_e[0, 0]), 0], [0, ti.exp(new_e[1, 1])]]) @ V.transpose()
        vc_s[p] += new_e.determinant() - e.determinant()
        # vc_s[p] += ti.log(new_F.determinant()) - ti.log(F_s[p].determinant())
        F_s[p] = new_F


    '''
    # G2P (water's part)
    for p in x_w:
        base = (x_w[p] * inv_dx - 0.5).cast(int)
        fx = x_w[p] * inv_dx - base.cast(float)
        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1.0) ** 2, 0.5 * (fx - 0.5) ** 2]
        new_v = ti.Vector.zero(float, 2)
        new_C = ti.Matrix.zero(float, 2, 2)
        for i, j in ti.static(ti.ndrange(3, 3)):
            dpos = ti.Vector([i, j]).cast(float) - fx
            g_v = grid_wv[base + ti.Vector([i, j])]
            weight = w[i][0] * w[j][1]
            new_v += weight * g_v
            new_C += 4 * inv_dx * weight * g_v.outer_product(dpos)
        J_w[p] = (1 + dt * new_C.trace()) * J_w[p]
        v_w[p], C_w[p] = new_v, new_C
        x_w[p] += dt * v_w[p]
    '''

@ti.kernel
def initialize():
    '''
    for i in range(n_w_particles):
        x_w[i] = [ti.random() * 0.5 + 0.3 + 0.10, ti.random() * 0.5 + 0.05 + 0.32]
        v_w[i] = ti.Matrix([0, 0])
        J_w[i] = 1
    '''
    for i in range(n_s_particles):
        x_s[i] = [ti.random() * 0.2 + 0.3 + 0.10, ti.random() * 0.5]
        v_s[i] = ti.Matrix([0, 0])
        F_s[i] = ti.Matrix([[1, 0], [0, 1]])
        c_C0[i] = 0.5
        vc_s[i] = 0
        alpha_s[i] = 0.3

initialize()
gui = ti.GUI("Test", res = 512, background_color = 0x112F41)
while not gui.get_event(ti.GUI.ESCAPE, ti.GUI.EXIT):
    for s in range(50):
        substep()
    # gui.circles(x_w.to_numpy(), radius=1.5, color = 0x068587)
    gui.circles(x_s.to_numpy(), radius = 1.5, color = 0x855E42)
    gui.show()
