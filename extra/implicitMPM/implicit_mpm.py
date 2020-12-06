import numpy as np
import taichi as ti

ti.init(arch = ti.cpu)

MATERIAL_PHASE_FLUID = 0
MATERIAL_PHASE_SOLID = 1

@ti.data_oriented
class IMPLICIT_MPM:
    def __init__(self,
    category = MATERIAL_PHASE_SOLID, dim = 2, dt = 0.00004, n_particles = 5000,  n_grid = 128, gravity = 9.8, gravity_dim = 1,
    p_rho = 1.0, E = 50000, nu = 0.4,
    newton_max_iterations = 1, newton_tolerance = 1e-2, line_search = True, linear_solve_tolerance_scale = 1,
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
        self.gravity = ti.Vector([(-gravity if i == gravity_dim else 0) for i in range(dim)])

        self.E, self.nu = E, nu # Young's modulus and Poisson's ratio
        self.mu_0, self.lambda_0 = E / (2 * (1 + nu)), E * nu / ((1+nu) * (1 - 2 * nu)) # Lame parameters

        # newton solver parameters        
        self.newton_max_iterations = newton_max_iterations
        self.newton_tolerance = newton_tolerance
        self.line_search = line_search
        self.linear_solve_tolerance_scale = linear_solve_tolerance_scale

        # linear solver parameters 
        self.linear_max_iterations = linear_max_iterations
        self.linear_tolerance = linear_tolerance

        self.cfl = cfl
        self.ppc = ppc

        self.real = real
        self.neighbour = (3, ) * dim
        self.bound = 3
        self.n_nodes = self.n_grid ** dim

        self.ignore_collision = False

        self.x = ti.Vector.field(dim, dtype = real, shape = n_particles) # position
        self.v = ti.Vector.field(dim, dtype = real, shape = n_particles) # velocity
        self.C = ti.Matrix.field(dim, dim, dtype = real, shape = n_particles) # affine velocity matrix
        self.F = ti.Matrix.field(dim, dim, dtype = real, shape = n_particles) # deformation gradient, i.e. strain
        self.old_F = ti.Matrix.field(dim, dim, dtype = real, shape = n_particles)

        self.grid_v = ti.Vector.field(dim, dtype = real) # grid node momentum/velocity
        self.grid_m = ti.field(dtype = real) # grid node mass

        block_size = 16
        indices = ti.ijk if self.dim == 3 else ti.ij
        self.grid = ti.root.pointer(indices, [self.n_grid // block_size])
        self.grid.dense(
            indices, block_size).place(self.grid_v, self.grid_m)

        # data of Newton's method
        self.mass_matrix = ti.field(dtype = real)
        self.dv = ti.Vector.field(dim, dtype = real) # dv = v(n+1) - v(n), Newton is formed from g(dv)=0
        self.vn = ti.Vector.field(dim, dtype = real)
        self.step_direction = ti.Vector.field(dim, dtype = real)
        self.residual = ti.Vector.field(dim, dtype = real)
        if ti.static(line_search):
            self.dv0 = ti.Vector.field(dim, dtype = real) # dv of last iteration, for line search only
        
        chip_size = 16
        self.newton_data = ti.root.pointer(ti.i, [self.n_nodes // chip_size])
        self.newton_data.dense(
            ti.i, chip_size).place(self.mass_matrix, self.dv, self.vn, self.step_direction, self.residual)
        if ti.static(line_search):
            self.newton_data.dense(ti.i, chip_size).place(self.dv0)

    @ti.func
    def idx(self, I):
        return sum([I[i] * self.n_grid ** i for i in range(self.dim)])

    @ti.func
    def node(self, p):
        return ti.Vector([(p % (self.n_grid ** (i + 1))) // (self.n_grid ** i) for i in range(self.dim)])

    @ti.kernel
    def copy(self, target : ti.template(), source : ti.template()):
        for I in ti.grouped(source):
            target[I] = source[I]

    @ti.kernel
    def scaledCopy(self, target : ti.template(), source : ti.template(), scale : ti.f32, scaled : ti.template()):
        for I in ti.grouped(source):
            target[I] = source[I] + scale * scaled[I]

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
 
    def reinitialize(self):
        self.grid.deactivate_all()
        self.newton_data.deactivate_all()

    @ti.kernel
    def particlesToGrid(self):
        for p in self.x:
            Xp = self.x[p] * self.inv_dx
            base = int(Xp - 0.5)
            fx = Xp - base
            w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1)**2, 0.5 * (fx - 0.5)**2] # Quadratic kernels  [http://mpm.graphics   Eqn. 123, with x=fx, fx-1,fx-2]            
            stress = (-self.dt * self.p_vol * 4 * self.inv_dx * self.inv_dx) * self.dpsi_dF(self.F[p]) * self.F[p].transpose()
            affine = self.p_mass * self.C[p] + stress
            # affine = self.p_mass * self.C[p]
            
            for offset in ti.static(ti.grouped(ti.ndrange(*self.neighbour))):
                dpos = (offset - fx) * self.dx
                weight = self.real(1)
                for i in ti.static(range(self.dim)):
                    weight *= w[offset[i]][i]
                
                self.grid_v[base + offset] += weight * (self.p_mass * self.v[p] + affine @ dpos)
                self.grid_m[base + offset] += weight * self.p_mass

        for I in ti.grouped(self.grid_m):
            if self.grid_m[I] > 0:
                self.grid_v[I] /= self.grid_m[I] # momentum to velocity

    @ti.kernel
    def buildMassMatrix(self):
        for I in ti.grouped(self.grid_m):
            mass = self.grid_m[I]
            if mass > 0:
                self.mass_matrix[self.idx(I)] = mass

    @ti.kernel
    def buildInitialDvAndVnForNewton(self):
        for I in ti.grouped(self.grid_m):
            if (self.grid_m[I] > 0):
                node_id = self.idx(I)
                if ti.static(not self.ignore_collision):
                    cond = I < self.bound or I > self.n_grid - self.bound
                    self.dv[node_id] = 0 if cond else self.gravity * self.dt
                else:
                    self.dv[node_id] = self.gravity * self.dt # Newton initial guess for non-collided nodes
                self.vn[node_id] = self.grid_v[I]

    @ti.kernel
    def backupStrain(self):
        for p in self.F:
            self.old_F[p] = self.F[p]

    @ti.kernel
    def restoreStrain(self):
        for p in self.F:
            self.F[p] = self.old_F[p]

    @ti.kernel
    def constructNewVelocityFromNewtonResult(self):
        for I in ti.grouped(self.grid_m):
            if self.grid_m[I] > 0:
                self.grid_v[I] += self.dv[self.idx(I)]

    @ti.kernel
    def totalEnergy(self) -> ti.f32:
        result = self.real(0)
        for p in self.F:
            result += self.psi(self.F[p]) * self.p_vol * self.F[p].determinant() # gathered from particles

        # inertia part
        for I in self.dv:
            m = self.mass_matrix[I]
            dv = self.dv[I]
            result += m * dv.dot(dv) / 2

        # gravity part
        for I in self.dv:
            m = self.mass_matrix[I]
            dv = self.dv[I]
            result -= self.dt * m * self.gravity.dot(dv)
        
        return result

    @ti.kernel
    def computeResidual(self):
        for I in self.dv:
            self.residual[I] = self.dt * self.mass_matrix[I] * self.gravity

        for I in self.dv:
            self.residual[I] -= self.mass_matrix[I] * self.dv[I]

        ti.block_dim(self.n_grid)
        for p in self.x:
            Xp = self.x[p] * self.inv_dx
            base = int(Xp - 0.5)
            fx = Xp - base
            w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1)**2, 0.5 * (fx - 0.5)**2]            
            # new_V = ti.zero(self.v[p])
            new_C = ti.zero(self.C[p])
            for offset in ti.static(ti.grouped(ti.ndrange(*self.neighbour))):
                dpos = (offset - fx) * self.dx
                weight = 1.0
                for i in ti.static(range(self.dim)):
                    weight *= w[offset[i]][i]

                g_v = self.grid_v[base + offset] + self.dv[self.idx(base + offset)]
                # new_V += weight * g_v
                new_C += 4 * self.inv_dx * weight * g_v.outer_product(dpos)

            # self.v[p] = new_V
            # self.C[p] = new_C
            # self.x[p] += self.dt * self.v[p]
            F = (ti.Matrix.identity(self.real, self.dim) + self.dt * new_C) @ self.F[p]
            stress = (-self.dt * self.p_vol * 4 * self.inv_dx * self.inv_dx) * self.dpsi_dF(F) * F.transpose()

            for offset in ti.static(ti.grouped(ti.ndrange(*self.neighbour))):
                dpos = (offset - fx) * self.dx
                weight = 1.0
                for i in ti.static(range(self.dim)):
                    weight *= w[offset[i]][i]

                force = weight * stress @ dpos
                self.residual[self.idx(base + offset)] += self.dt * force

        # project()
        for p in self.dv:
            I = self.node(p)
            cond = any(I < self.bound) or any(I > self.n_grid - self.bound)
            if cond: self.dv[p] = ti.Vector.zero(self.real, self.dim)


    @ti.kernel
    def computeNorm(self) ->ti.f32:
        norm_sq = self.real(0)
        for I in self.dv:
            mass = self.mass_matrix[I]
            residual = self.residual[I]
            if mass > 0:
                norm_sq += residual.dot(residual) / mass
        return ti.sqrt(norm_sq)

    @ti.kernel
    def updateState(self):
        x = 0

    def backwardEulerStep(self): # on the assumption that collision is ignored
        self.buildMassMatrix()
        self.buildInitialDvAndVnForNewton()
        # Which should be called at the beginning of newton.
        self.backupStrain()

        self.newtonSolve()

        self.restoreStrain()
        self.constructNewVelocityFromNewtonResult()

    def newtonSolve(self):
        E0 = 0 # totalEnergy of last iteration, for line search only
        if ti.static(self.line_search):
            E0 = self.totalEnergy()
            self.copy(self.dv0, self.dv)

        for it in range(self.newton_max_iterations):
            # Mv^(n) - Mv^(n+1) + dt * f(x_n + dt v^(n+1)) + dt * Mg
            # -Mdv + dt * f(x_n + dt(v^n + dv)) + dt * Mg
            self.computeResidual()
            residual_norm = self.computeNorm()
            print(residual_norm)
            if residual_norm < self.newton_tolerance:
                break;

            linear_solve_relative_tolerance = ti.min(0.5, self.linear_solve_tolerance_scale * ti.sqrt(ti.max(residual_norm, self.newton_tolerance)));
            if ti.static(self.line_search):
                step_size, E = self.real(1), self.real(0)
                while True:
                    self.scaledCopy(self.dv, self.dv0, step_size, self.step_direction)
                    self.updateState()
                    E = self.totalEnergy()
                    step_size /= 2
                    if E < E0: break
                E0 = E
                self.copy(self.dv0, self.dv)


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
                weight = self.real(1)
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

    @ti.kernel
    def gridUpdateExplicit(self): # for test
        for I in ti.grouped(self.grid_m):
            self.grid_v[I] += self.dt * self.gravity
            cond = I < self.bound and self.grid_v[I] < 0 or I > self.n_grid - self.bound and self.grid_v[I] > 0
            self.grid_v[I] = 0 if cond else self.grid_v[I]


if __name__ == '__main__':
    solver = IMPLICIT_MPM()
    solver.init()

    gui = ti.GUI("Taichi MLS-MPM-128", res = 512, background_color = 0x112F41)
    while gui.running:
        for i in range(10):
            solver.reinitialize()
            solver.particlesToGrid()
            solver.gridUpdateExplicit()
            solver.backwardEulerStep()
            solver.gridToParticles()

        pos = solver.x.to_numpy()
        gui.circles(pos, radius=1.5, color=0x66ccff)
        gui.show()
