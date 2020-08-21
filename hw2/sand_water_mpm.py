import taichi as ti
import numpy as np
ti.init(arch=ti.gpu)

quality = 1
n_s_particles, n_w_particles, n_grid = 3000 * quality ** 2, 3000 * quality ** 2
dx, inv_dx = 1 / n_grid, float(n_grid)
dt = 1e-4 / quality
gravity = ti.Vector([0, 1])

# sand particle properties
x_s = ti.Vector.field(2, dtype = float, shape = n_s_particles) # position
v_s = ti.Vector.field(2, dtype = float, shape = n_s_particles) # velocity
C_s = ti.Matrix.field(2, 2, dtype = float, shape = n_s_particles) # affine velocity matrix
F_s = ti.Matrix.field(2, 2, dtype = float, shape = n_s_particles) # deformation gradient
c_C = ti.field(dtype = float, shape = n_s_particles) # cohesion and saturation
cC0 = ti.field(dtype = float, shape = n_s_particles) # initial cohesion (as maximum)

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
p_vol, s_rho, w_rho = (dx * 0.5) ** 2, 2, 2
s_mass, w_mass = p_vol * s_rho, p_vol * w_rho

k = 50, gamma = 3 # bulk modulus of water and gamma is a term that more stiffy penalizes large deviations from incompressibility

n, k_hat = 0.4, 0.2 # sand porosity and permeability

E_s, nu_s = 400, 0.4 # sand's Young's modulus and Poisson's ratio
mu_s, lambda_s = E_s / (2 * (1 + nu_s)), E_s * nu_s / ((1 + nu_s) * (1 - 2 * nu_s)) # sand's Lame parameters

a, b, c0, sC = -3.0, 0, 1e-2, 0.15
# The scalar function h_s is chosen so that the multiplier function is twice continuously differentiable
@ti.func
def h_s(z):
    if z < 0: return 1
    if z > 1: return 0
    return 1 - 10 * z ** 3 + 15 * z ** 4 - 6 * z ** 5

# multiplier
@ti.func
def h(e):
    u = (e[0, 0] + e[1, 1]) / ti.static(ti.sqrt(2))
    v = ti.abs(ti.Vector([e[0, 0] - u / ti.static(ti.sqrt(2)),
                          e[1, 1] - u / ti.static(ti.sqrt(2))
                         ]).norm())
    fe = c0 * (v ** 4) / (1 + v ** 3)
    if u + fe < a + sC: return 1
    if u + fe > b + sC: return 0
    return h_s((u + fe - a - sC) / (b - a))

@ti.kernel
def substep():
    # set zero initial state for both water/sand grid
    for i, j in grid_sm:
        grid_sv[i, j], grid_wv[i, j] = [0, 0], [0, 0]
        grid_sm[i, j], grid_wm[i, j] = 0, 0
        f_s[i, j], f_w[i, j] = [0, 0], [0, 0]

    # P2G (sand's part)
    for p in x_s:
        base = (x_s[p] * inv_dx - 0.5).cast(int)
        fx = x_s[p] * inv_dx - base.cast(float)
        # Quadratic kernels  [http://mpm.graphics   Eqn. 123, with x=fx, fx-1,fx-2]
        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]
        U, sig, V = ti.svd(F_s[p])
        inv_sig = sig.inverse()
        ln_sig = ti.log(sig)
        stress = U @ (2 * mu_s * inv_sig @ ln_sig +
                lambda_s * (ln_sig[0, 0] + ln_sig[1, 1]) * inv_sig) @ V.transpose() * h(ln_sig) # formula (25)
        stress = (-p_vol * 4 * inv_dx * inv_dx) * stress * F_s.transpose()
        affine = s_mass * C_s[p]
        for i, j in ti.static(ti.ndrange(3, 3)):
            offset = ti.Vector([i, j])
            dpos = (offset.cast(float) - fx) * dx
            weight = w[i][0] * w[j][1]
            grid_sv[base + offset] += weight * (s_mass * v_s[p] + affine @ dpos)
            grid_sm[base + offset] += weight * s_mass
            grid_sf[base + offset] += weight * stress @ dpos

    # P2G (water's part):
    for p in x_w:
        base = (x[p] * inv_dx - 0.5).cast(int)
        fx = x[p] * inv_dx - base.cast(float)
        # Quadratic kernels  [http://mpm.graphics   Eqn. 123, with x=fx, fx-1,fx-2]
        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]
        stress = k * (1 - 1 / (J_w ** gamma))
        stress = (-p_vol * 4 * inv_dx * inv_dx) * stress * J
        affine = w_mass * C_w[p]
        for i, j in ti.static(ti.ndrange(3, 3)):
            offset = ti.Vector([i, j])
            dpos = (offset.cast(float) - fx) * dx
            weight = w[i][0] * w[j][1]
            grid_wv[base + offset] += weight * (w_mass * v_w[p] + affine @ dpos)
            grid_wm[base + offset] += weight * w_mass
            grid_wf[base + offset] += weight * stress * dpos

    # Update Grids Momenta
    for i, j in grid_sm:
        if grid_sm[i, j] > 0:
            grid_sv[i, j] = (1 / grid_sm[i, j]) * grid_sv[i, j] # Momentum to velocity
            grid_sv[i, j] += (dt / grid_sm[i, j]) * (gravity + grid_sf[i, j]) # Update acceleration
        if grid_wm[i, j] > 0:
            grid_wv[i, j] = (1 / grid_wm[i, j]) * grid_wv[i, j]
            grid_wv[i, j] += (dt / grid_wm[i, j]) * (gravity + grid_wf[i, j])

        if grid_sm[i, j] > 0 and grid_wm[i, j] > 0:
            cE = (n * n * w_rho * gravity[1]) / k_hat
            p_s = cE * (grid_wv[i, j] - grid_sv[i, j])
            grid_sv[i, j] += p_s / grid_sm[i, j]
            grid_wv[i, j] -= p_s / grid_wm[i, j]

        if grid_sm[i, j] > 0:
            if i < 3 and grid_sv[i, j][0] < 0:          grid_sv[i, j][0] = 0 # Boundary conditions
            if i > n_grid - 3 and grid_sv[i, j][0] > 0: grid_sv[i, j][0] = 0
            if j < 3 and grid_sv[i, j][1] < 0:          grid_sv[i, j][1] = 0
            if j > n_grid - 3 and grid_sv[i, j][1] > 0: grid_sv[i, j][1] = 0

        if grid_wm[i, j] > 0:
            if i < 3 and grid_wv[i, j][0] < 0:          grid_wv[i, j][0] = 0 # Boundary conditions
            if i > n_grid - 3 and grid_wv[i, j][0] > 0: grid_wv[i, j][0] = 0
            if j < 3 and grid_wv[i, j][1] < 0:          grid_wv[i, j][1] = 0
            if j > n_grid - 3 and grid_wv[i, j][1] > 0: grid_wv[i, j][1] = 0

    # G2P (sand's part)
    for p in x_s:
        base = (x_s[p] * inv_dx - 0.5).cast(int)
        fx = x_s[p] * inv_dx - base.cast(float)
        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1.0) ** 2, 0.5 * (fx - 0.5) ** 2]

        F_s[p] = (ti.Matrix.identity(float, 2) + dt * C_s[p]) @ F_s[p]
        phi = 0
        for i, j in ti.static(ti.ndrange(3, 3)): # loop over 3x3 grid node neighborhood
            dpos = ti.Vector([i, j]).cast(float) - fx
            weight = w[i][0] * w[j][1]
            if grid_sm[i, j] > 0 and grid_wm[i, j] > 0:
                phi += weight # formula (24)
        c_C[p] = c_C0[p] * (1 - phi)


    # G2P (water's part)
    for p in x_w:
        base = (x_w[p] * inv_dx - 0.5).cast(int)
        fx = x_w[p] * inv_dx - base.cast(float)
        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1.0) ** 2, 0.5 * (fx - 0.5) ** 2]

        J_w[p] = (1 + dt * (C_w[p][0, 0] + C_w[p][1, 1])) * J_w[p]

