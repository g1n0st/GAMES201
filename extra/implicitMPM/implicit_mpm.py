import numpy as np
import taichi as ti

ti.init(arch = ti.cpu)

MATERIAL_PHASE_FLUID = 0
MATERIAL_PHASE_SOLID = 1

@ti.data_oriented
class IMPLICIT_MPM:
    def __init__(self,
    category = MATERIAL_PHASE_SOLID, dim = 2, dt = 0.00004, n_particles = 5000,  n_grid = 128, gravity = 9.8,
    p_rho = 1.0, E = 50000, nu = 0.4,
    newton_max_iterations = 1, newton_tolerance = 1e-2, line_search = True,
    linear_max_iterations = 10000, linear_tolerance = 1e-6,
    cfl = 0.4, ppc = 16, 
    real = float
    ):
        self.category = category
        self.dim = dim
        self.dt = dt
        self.n_particles = n_particles
        self.n_grid = n_grid
        self.dx = 1 / n_grid
        self.inv_dx = float(n_grid)
        self.p_rho = p_rho
        self.p_vol = (self.dx * 0.5) ** self.dim
        self.p_mass = self.p_vol * self.p_rho
        self.gravity = gravity

        self.E, self.nu = E, nu # Young's modulus and Poisson's ratio
        self.mu_0, self.lambda_0 = E / (2 * (1 + nu)), E * nu / ((1+nu) * (1 - 2 * nu)) # Lame parameters

        # newton solver parameters        
        self.newton_max_iterations = newton_max_iterations
        self.newton_tolerance = newton_tolerance
        self.line_search = line_search

        # linear solver parameters 
        self.linear_max_iterations = linear_max_iterations
        self.linear_tolerance = linear_tolerance

        self.cfl = cfl
        self.ppc = ppc

        self.real = real
        self.neighbour = (3, ) * dim
        self.bound = 3

        self.x = ti.Vector.field(dim, dtype = real, shape = n_particles) # position
        self.v = ti.Vector.field(dim, dtype = real, shape = n_particles) # velocity
        self.C = ti.Matrix.field(dim, dim, dtype = real, shape = n_particles) # affine velocity matrix
        if (ti.static(self.category == MATERIAL_PHASE_SOLID)):
            self.F = ti.Matrix.field(dim, dim, dtype = real, shape = n_particles) # deformation gradient [SOLID PHASE]
        else:
            self.J = ti.field(dtype = real, shape = n_particles) # ratio of volume increase [FLUID PHASE]

        self.grid_v = ti.Vector.field(dim, dtype = real) # grid node momentum/velocity
        self.grid_m = ti.field(dtype = real) # grid node mass

        block_size = 16
        indices = ti.ijk if self.dim == 3 else ti.ij
        self.grid = ti.root.pointer(indices, [self.n_grid // block_size]).dense(
            indices, block_size).place(self.grid_v, self.grid_m)
    
    # TODO: Abstract as general stress classes
    @ti.func
    def psi(self, F): # strain energy density function Ψ(F)
        U, sig, V = ti.svd(F)
        
        # fixed corotated model, you can replace it with any constitutive model
        return self.mu_0 * sum([(sig[i, i] - 1) ** 2 for i in range(self.dim)]) + self.lambda_0 / 2 * (F.determinant() - 1) ** 2
    
    @ti.func
    def dpsi_dF(self, F): # first Piola-Kirchoff stress P(F), i.e. ∂Ψ/∂F
        U, sig, V = ti.svd(F)
        J = F.determinant()
        R = U @ V.transpose()
        return 2 * self.mu_0 * (F - R) + self.lambda_0 * (J - 1) * J * F.inverse().transpose()

    @ti.kernel
    def particlesToGrid(self):
        # reinitialize
        for I in ti.grouped(self.grid_m):
            self.grid_v[I] = ti.zero(self.grid_v[I])
            self.grid_m[I] = 0

        ti.block_dim(self.n_grid)
        for p in self.x:
            Xp = self.x[p] * self.inv_dx
            base = int(Xp - 0.5)
            fx = Xp - base
            w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1)**2, 0.5 * (fx - 0.5)**2] # Quadratic kernels  [http://mpm.graphics   Eqn. 123, with x=fx, fx-1,fx-2]            
            stress = (-self.dt * self.p_vol * 4 * self.inv_dx * self.inv_dx) * self.dpsi_dF(self.F[p]) * self.F[p].transpose()
            affine = self.p_mass * self.C[p] + stress
            
            for offset in ti.static(ti.grouped(ti.ndrange(*self.neighbour))):
                dpos = (offset - fx) * self.dx
                weight = self.real(1.0)
                for i in ti.static(range(self.dim)):
                    weight *= w[offset[i]][i]
                
                self.grid_v[base + offset] += weight * (self.p_mass * self.v[p] + affine @ dpos)
                self.grid_m[base + offset] += weight * self.p_mass

    @ti.kernel
    def gridUpdate(self):
        for I in ti.grouped(self.grid_m):
            if self.grid_m[I] > 0:
                self.grid_v[I] /= self.grid_m[I]
            self.grid_v[I][1] -= self.dt * self.gravity
            cond = I < self.bound and self.grid_v[I] < 0 or I > self.n_grid - self.bound and self.grid_v[I] > 0
            self.grid_v[I] = 0 if cond else self.grid_v[I]

    @ti.kernel
    def gridToParticles(self):
        ti.block_dim(self.n_grid)
        for p in self.x:
            Xp = self.x[p] * self.inv_dx
            base = int(Xp - 0.5)
            fx = Xp - base
            w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1)**2, 0.5 * (fx - 0.5)**2] # Quadratic kernels  [http://mpm.graphics   Eqn. 123, with x=fx, fx-1,fx-2]
            new_V = ti.zero(self.v[p])
            new_C = ti.zero(self.C[p])
            for offset in ti.static(ti.grouped(ti.ndrange(*self.neighbour))):
                dpos = (offset - fx) * self.dx
                weight = 1.0
                for i in ti.static(range(self.dim)):
                    weight *= w[offset[i]][i]

                g_v = self.grid_v[base + offset]
                new_V += weight * g_v
                new_C += 4 * self.inv_dx * weight * g_v.outer_product(dpos)

            self.v[p] = new_V
            self.C[p] = new_C
            self.F[p] = (ti.Matrix.identity(self.real, self.dim) + self.dt * self.C[p]) @ self.F[p] # F' = (I+dt * grad v)F
            self.x[p] += self.dt * self.v[p]

    
    @ti.kernel
    def init(self):
        for i in range(self.n_particles):
            self.x[i] = ti.Vector([ti.random() for i in range(self.dim)]) * 0.4 + 0.15
            self.F[i] = ti.Matrix.identity(self.real, self.dim)


if __name__ == '__main__':
    solver = IMPLICIT_MPM()
    solver.init()

    gui = ti.GUI("Taichi MLS-MPM-128", res=512, background_color=0x112F41)
    while gui.running:
        for i in range(100):
            solver.particlesToGrid()
            solver.gridUpdate()
            solver.gridToParticles()

        pos = solver.x.to_numpy()
        gui.circles(pos, radius=1.5, color=0x66ccff)
        gui.show()
