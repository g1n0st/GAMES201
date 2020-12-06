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
    linear_solve_max_iterations = 10000, linear_solve_tolerance = 1e-6,
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
        self.linear_solve_max_iterations = linear_solve_max_iterations
        self.linear_solve_tolerance = linear_solve_tolerance

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
        self.old_F = ti.Matrix.field(dim, dim, dtype = real, shape = n_particles) # for backup/restore F

        # These should be updated everytime a new SVD is performed to F
        if ti.static(dim == 2):
            self.psi0 = ti.field(dtype = real, shape = n_particles)   # d_PsiHat_d_sigma0
            self.psi1 = ti.field(dtype = real, shape = n_particles)   # d_PsiHat_d_sigma1
            self.psi00 = ti.field(dtype = real, shape = n_particles)  # d^2_PsiHat_d_sigma0_d_sigma0
            self.psi01 = ti.field(dtype = real, shape = n_particles)  # d^2_PsiHat_d_sigma0_d_sigma1
            self.psi11 = ti.field(dtype = real, shape = n_particles)  # d^2_PsiHat_d_sigma1_d_sigma1
            self.m01 = ti.field(dtype = real, shape = n_particles)    # (psi0-psi1)/(sigma0-sigma1), usually can be computed robustly
            self.p01 = ti.field(dtype = real, shape = n_particles)    # (psi0+psi1)/(sigma0+sigma1), need to clamp bottom with 1e-6
            self.Aij = ti.Matrix.field(dim, dim, dtype = real, shape = n_particles) 
            self.B01 = ti.Matrix.field(2, 2, dtype = real, shape = n_particles)
        if ti.static(dim == 3):
            psi0 = ti.field(dtype = real, shape = n_particles)  # d_PsiHat_d_sigma0
            psi1 = ti.field(dtype = real, shape = n_particles)  # d_PsiHat_d_sigma1
            psi2 = ti.field(dtype = real, shape = n_particles)  # d_PsiHat_d_sigma2
            psi00 = ti.field(dtype = real, shape = n_particles) # d^2_PsiHat_d_sigma0_d_sigma0
            psi11 = ti.field(dtype = real, shape = n_particles) # d^2_PsiHat_d_sigma1_d_sigma1
            psi22 = ti.field(dtype = real, shape = n_particles) # d^2_PsiHat_d_sigma2_d_sigma2
            psi01 = ti.field(dtype = real, shape = n_particles) # d^2_PsiHat_d_sigma0_d_sigma1
            psi02 = ti.field(dtype = real, shape = n_particles) # d^2_PsiHat_d_sigma0_d_sigma2
            psi12 = ti.field(dtype = real, shape = n_particles) # d^2_PsiHat_d_sigma1_d_sigma2

            m01 = ti.field(dtype = real, shape = n_particles)   # (psi0-psi1)/(sigma0-sigma1), usually can be computed robustly
            p01 = ti.field(dtype = real, shape = n_particles)   # (psi0+psi1)/(sigma0+sigma1), need to clamp bottom with 1e-6
            m02 = ti.field(dtype = real, shape = n_particles)   # (psi0-psi2)/(sigma0-sigma2), usually can be computed robustly
            p02 = ti.field(dtype = real, shape = n_particles)   # (psi0+psi2)/(sigma0+sigma2), need to clamp bottom with 1e-6
            m12 = ti.field(dtype = real, shape = n_particles)   # (psi1-psi2)/(sigma1-sigma2), usually can be computed robustly
            p12 = ti.field(dtype = real, shape = n_particles)   # (psi1+psi2)/(sigma1+sigma2), need to clamp bottom with 1e-6
            Aij = ti.Matrix.field(dim, dim, dtype = real, shape = n_particles) 
            B01 = ti.Matrix.field(2, 2, dtype = real, shape = n_particles)
            B12 = ti.Matrix.field(2, 2, dtype = real, shape = n_particles)
            B20 = ti.Matrix.field(2, 2, dtype = real, shape = n_particles)

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
        self.residual = ti.Vector.field(dim, dtype = real)
        if ti.static(line_search):
            self.dv0 = ti.Vector.field(dim, dtype = real) # dv of last iteration, for line search only

        # data of Linear Solver, i.e. Conjugate Gradient
        # All notations adopted from Wikipedia, q denotes A*p in general
        self.r = ti.Vector.field(dim, dtype = real)
        self.p = ti.Vector.field(dim, dtype = real)
        self.q = ti.Vector.field(dim, dtype = real)
        self.temp = ti.Vector.field(dim, dtype = real)
        self.step_direction = ti.Vector.field(dim, dtype = real)

        # scratch data for calculate differential of F
        self.scratch_xp = ti.Vector.field(dim, dtype = real, shape = n_particles)
        self.scratch_vp = ti.Vector.field(dim, dtype = real, shape = n_particles)
        self.scratch_gradV = ti.Matrix.field(dim, dim, dtype = real, shape = n_particles)
        self.scratch_stress = ti.Matrix.field(dim, dim, dtype = real, shape = n_particles)
        
        chip_size = 16
        self.newton_data = ti.root.pointer(ti.i, [self.n_nodes // chip_size])
        self.newton_data.dense(
            ti.i, chip_size).place(self.mass_matrix, self.dv, self.vn, self.residual)
        if ti.static(line_search):
            self.newton_data.dense(ti.i, chip_size).place(self.dv0)

        self.linear_solver_data = ti.root.pointer(ti.i, [self.n_nodes // chip_size])
        self.linear_solver_data.dense(
            ti.i, chip_size).place(self.r, self.p, self.q, self.temp, self.step_direction)

    '''
    Math Library
    '''
    @ti.func
    def clamp_small_magnitude(self, x, eps):
        result = self.real(0)
        if x < -eps: result = x
        elif x < 0: result = -eps
        elif x < eps: result = eps
        else: result = x
        return result

    @ti.func
    def makePD(self, M : ti.template()):
        U, sigma, V = ti.svd(M)
        if sigma[0, 0] >= 0: return
        D = ti.Vector(self.dim, self.real)
        for i in range(self.dim):
            if sigma[i, i] < 0: D[i] = 0
            else: D[i] = sigma[i, i]
        
        M = sigma @ D @ sigma.transpose()

    @ti.func
    def makePD2d(self, M : ti.template()):
        a = M[0, 0]
        b = (M[0, 1] + M[1, 0]) / 2
        d = M[1, 1]
        
        b2 = b * b
        D = a * d - b2
        T_div_2 = (a + d) / 2
        sqrtTT4D = ti.sqrt(ti.abs(T_div_2 * T_div_2 - D))
        L2 = T_div_2 - sqrtTT4D
        if L2 < 0.0:
            L1 = T_div_2 + sqrtTT4D
            if L1 < 0.0: M = ti.zero(M)
            else:
                if b2 == 0: M = ti.Matrix([[L1, 0], [0, 0]])
                else:
                    L1md = L1 - d
                    L1md_div_L1 = L1md / L1
                    M = ti.Matrix([[L1md_div_L1 * L1md, b * L1md_div_L1], [b * L1md_div_L1, b2 / L1]])

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

    # B = dPdF(Sigma) : A
    @ti.func
    def dPdFOfSigmaContractProjected(self, p, A, B : ti.template()):
        if ti.static(self.dim == 2):
            B[0, 0] = self.Aij[p][0, 0] * A[0, 0] + self.Aij[p][0, 1] * A[1, 1]
            B[1, 1] = self.Aij[p][1, 0] * A[0, 0] + self.Aij[p][1, 1] * A[1, 1]
            B[0, 1] = self.B01[p][0, 0] * A[0, 1] + self.B01[p][0, 1] * A[1, 0]
            B[1, 0] = self.B01[p][1, 0] * A[0, 1] + self.B01[p][1, 1] * A[1, 0]
        if ti.static(self.dim == 3):
            B[0, 0] = self.Aij[p][0, 0] * A[0, 0] + self.Aij[p][0, 1] * A[1, 1] + self.Aij[p][0, 2] * A[2, 2]
            B[1, 1] = self.Aij[p][1, 0] * A[0, 0] + self.Aij[p][1, 1] * A[1, 1] + self.Aij[p][1, 2] * A[2, 2]
            B[2, 2] = self.Aij[p][2, 0] * A[0, 0] + self.Aij[p][2, 1] * A[1, 1] + self.Aij[p][2, 2] * A[2, 2]
            B[0, 1] = self.B01[p][0, 0] * A[0, 1] + self.B01[p][0, 1] * A[1, 0]
            B[1, 0] = self.B01[p][1, 0] * A[0, 1] + self.B01[p][1, 1] * A[1, 0]
            B[0, 2] = self.B20[p][0, 0] * A[0, 2] + self.B20[p][0, 1] * A[2, 0]
            B[2, 0] = self.B20[p][1, 0] * A[0, 2] + self.B20[p][1, 1] * A[2, 0]
            B[1, 2] = self.B12[p][0, 0] * A[1, 2] + self.B12[p][0, 1] * A[2, 1]
            B[2, 1] = self.B12[p][1, 0] * A[1, 2] + self.B12[p][1, 1] * A[2, 1]
        
    @ti.func
    def firstPiolaDifferential(self, p, F, dF, dP : ti.template()):
        U, sig, V = ti.svd(F)
        D = U.transpose() @ dF @ V
        K = ti.Matrix.zero(self.real, self.dim, self.dim)
        self.dPdFOfSigmaContractProjected(p, D, K)
        dP = U @ K @ V.transpose()

    @ti.func
    def reinitializeIsotropicHelper(self, p):
        if ti.static(self.dim == 2):
            self.psi0[p] = 0 # d_PsiHat_d_sigma0
            self.psi1[p] = 0 # d_PsiHat_d_sigma1
            self.psi00[p] = 0 # d^2_PsiHat_d_sigma0_d_sigma0
            self.psi01[p] = 0 # d^2_PsiHat_d_sigma0_d_sigma1
            self.psi11[p] = 0 # d^2_PsiHat_d_sigma1_d_sigma1
            self.m01[p] = 0 # (psi0-psi1)/(sigma0-sigma1), usually can be computed robustly
            self.p01[p] = 0 # (psi0+psi1)/(sigma0+sigma1), need to clamp bottom with 1e-6
            self.Aij[p] = ti.zero(self.Aij[p])
            self.B01[p] = ti.zero(self.B01[p])
        if ti.static(self.dim == 3):
            self.psi0[p] = 0 # d_PsiHat_d_sigma0
            self.psi1[p] = 0 # d_PsiHat_d_sigma1
            self.psi2[p] = 0 # d_PsiHat_d_sigma2
            self.psi00[p] = 0 # d^2_PsiHat_d_sigma0_d_sigma0
            self.psi11[p] = 0 # d^2_PsiHat_d_sigma1_d_sigma1
            self.psi22[p] = 0 # d^2_PsiHat_d_sigma2_d_sigma2
            self.psi01[p] = 0 # d^2_PsiHat_d_sigma0_d_sigma1
            self.psi02[p] = 0 # d^2_PsiHat_d_sigma0_d_sigma2
            self.psi12[p] = 0 # d^2_PsiHat_d_sigma1_d_sigma2

            self.m01[p] = 0 # (psi0-psi1)/(sigma0-sigma1), usually can be computed robustly
            self.p01[p] = 0 # (psi0+psi1)/(sigma0+sigma1), need to clamp bottom with 1e-6
            self.m02[p] = 0 # (psi0-psi2)/(sigma0-sigma2), usually can be computed robustly
            self.p02[p] = 0 # (psi0+psi2)/(sigma0+sigma2), need to clamp bottom with 1e-6
            self.m12[p] = 0 # (psi1-psi2)/(sigma1-sigma2), usually can be computed robustly
            self.p12[p] = 0 # (psi1+psi2)/(sigma1+sigma2), need to clamp bottom with 1e-6
            self.Aij[p] = ti.zero(self.Aij[p])
            self.B01[p] = ti.zero(self.B01[p])
            self.B12[p] = ti.zero(self.B12[p])
            self.B20[p] = ti.zero(self.B20[p])

    @ti.func
    def updateIsotropicHelper(self, p, F):
        self.reinitializeIsotropicHelper(p)
        if ti.static(self.dim == 2):
            U, sigma, V = ti.svd(F)
            J = sigma[0, 0] * sigma[1, 1]
            _2mu = self.mu_0 * 2
            _lambda = self.lambda_0 * (J - 1)
            Sprod = ti.Vector([sigma[1, 1], sigma[0, 0]])
            self.psi0[p] = _2mu * (sigma[0, 0] - 1) + _lambda * Sprod[0]
            self.psi1[p] = _2mu * (sigma[1, 1] - 1) + _lambda * Sprod[1]
            self.psi00[p] = _2mu + self.lambda_0 * Sprod[0] * Sprod[0]
            self.psi11[p] = _2mu + self.lambda_0 * Sprod[1] * Sprod[1]
            self.psi01[p] = _lambda + self.lambda_0 * Sprod[0] * Sprod[1]

            # (psi0-psi1)/(sigma0-sigma1)
            self.m01[p] = _2mu - _lambda

            # (psi0+psi1)/(sigma0+sigma1)
            self.p01[p] = (self.psi0[p] + self.psi1[p]) / self.clamp_small_magnitude(sigma[0, 0] + sigma[1, 1], 1e-6)

            self.Aij[p] = ti.Matrix(
                [[self.psi00[p], self.psi01[p]], 
                 [self.psi01[p], self.psi11[p]]])
            self.B01[p] = ti.Matrix(
                [[(self.m01[p] + self.p01[p]) * 0.5, (self.m01[p] - self.p01[p]) * 0.5], 
                 [(self.m01[p] - self.p01[p]) * 0.5, (self.m01[p] + self.p01[p]) * 0.5]])
            
            # proj A
            self.makePD(self.Aij[p])
            # proj B
            self.makePD2d(self.B01[p])
        if ti.static(self.dim == 3):
            U, sigma, V = ti.svd(F)
            J = sigma[0, 0] * sigma[1, 1] * sigma[2, 2]
            _2mu = self.mu_0 * 2
            _lambda = self.lambda_0* (J - 1)
            Sprod = ti.Vector([sigma[1, 1] * sigma[2, 2], sigma[0, 0] * sigma[2, 2], sigma[0, 0] * sigma[1, 1]])
            self.psi0[p] = _2mu * (sigma[0, 0] - 1) + _lambda * Sprod[0]
            self.psi1[p] = _2mu * (sigma[1, 1] - 1) + _lambda * Sprod[1]
            self.psi2[p] = _2mu * (sigma[2, 2] - 1) + _lambda * Sprod[2]
            self.psi00[p] = _2mu +self.lambda_0* Sprod[0] * Sprod[0]
            self.psi11[p] = _2mu +self.lambda_0* Sprod[1] * Sprod[1]
            self.psi22[p] = _2mu +self.lambda_0* Sprod[2] * Sprod[2]
            self.psi01[p] = _lambda * sigma[2, 2] +self.lambda_0* Sprod[0] * Sprod[1]
            self.psi02[p] = _lambda * sigma[1, 1] +self.lambda_0* Sprod[0] * Sprod[2]
            self.psi12[p] = _lambda * sigma[0, 0] +self.lambda_0* Sprod[1] * Sprod[2]

            # (psiA-psiB)/(sigmaA-sigmaB)
            self.m01[p] = _2mu - _lambda * sigma[2, 2] # i[p] = 0
            self.m02[p] = _2mu - _lambda * sigma[1, 1] # i[p] = 2
            self.m12[p] = _2mu - _lambda * sigma[0, 0] # i[p] = 1

            # (psiA+psiB)/(sigmaA+sigmaB)
            self.p01[p] = (self.psi0[p] + self.psi1[p]) / self.clamp_small_magnitude(sigma[0, 0] + sigma[1, 1], 1e-6)
            self.p02[p] = (self.psi0[p] + self.psi2[p]) / self.clamp_small_magnitude(sigma[0, 0] + sigma[2, 2], 1e-6)
            self.p12[p] = (self.psi1[p] + self.psi2[p]) / self.clamp_small_magnitude(sigma[1, 1] + sigma[2, 2], 1e-6)

            self.Aij[p] = ti.Matrix([
                [self.psi00[p], self.psi01[p], self.psi02[p]],
                [self.psi01[p], self.psi11[p], self.psi12[p]],
                [self.psi02[p], self.psi12[p], self.psi22[p]]])
            self.B01[p] = ti.matrix([
                [(self.m01[p] + self.p01[p]) * 0.5, (self.m01[p] - self.p01[p]) * 0.5],
                [(self.m01[p] - self.p01[p]) * 0.5, (self.m01[p] + self.p01[p]) * 0.5]])
            self.B12[p] = ti.matrix([
                [(self.m12[p] + self.p12[p]) * 0.5, (self.m12[p] - self.p12[p]) * 0.5],
                [(self.m12[p] - self.p12[p]) * 0.5, (self.m12[p] + self.p12[p]) * 0.5]])
            self.B20[p] = ti.matrix([
                [(self.m02[p] + self.p02[p]) * 0.5, (self.m02[p] - self.p02[p]) * 0.5],
                [(self.m02[p] - self.p02[p]) * 0.5, (self.m02[p] + self.p02[p]) * 0.5]])

            # proj A
            self.makePD(self.Aij[p])
            # proj B
            self.makePD2d(self.B01[p])
            self.makePD2d(self.B12[p])
            self.makePD2d(self.B20[p])
 
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
            stress = (-self.p_vol * 4 * self.inv_dx * self.inv_dx) * self.dpsi_dF(self.F[p]) @ self.F[p].transpose()
            affine = self.p_mass * self.C[p] + self.dt * stress
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
            new_C = ti.zero(self.C[p])
            for offset in ti.static(ti.grouped(ti.ndrange(*self.neighbour))):
                dpos = (offset - fx) * self.dx
                weight = 1.0
                for i in ti.static(range(self.dim)):
                    weight *= w[offset[i]][i]

                g_v = self.grid_v[base + offset] + self.dv[self.idx(base + offset)]
                new_C += 4 * self.inv_dx * weight * g_v.outer_product(dpos)

            F = (ti.Matrix.identity(self.real, self.dim) + self.dt * new_C) @ self.old_F[p]
            stress = (-self.p_vol * 4 * self.inv_dx * self.inv_dx) * self.dpsi_dF(F) @ F.transpose()

            for offset in ti.static(ti.grouped(ti.ndrange(*self.neighbour))):
                dpos = (offset - fx) * self.dx
                weight = 1.0
                for i in ti.static(range(self.dim)):
                    weight *= w[offset[i]][i]

                force = weight * stress @ dpos
                self.residual[self.idx(base + offset)] += self.dt * force

        self.project(self.dv)

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
        ti.block_dim(self.n_grid)
        for p in self.x:
            Xp = self.x[p] * self.inv_dx
            base = int(Xp - 0.5)
            fx = Xp - base
            w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1)**2, 0.5 * (fx - 0.5)**2]            
            new_C = ti.zero(self.C[p])
            for offset in ti.static(ti.grouped(ti.ndrange(*self.neighbour))):
                dpos = (offset - fx) * self.dx
                weight = 1.0
                for i in ti.static(range(self.dim)):
                    weight *= w[offset[i]][i]

                g_v = self.grid_v[base + offset] + self.dv[self.idx(base + offset)]
                new_C += 4 * self.inv_dx * weight * g_v.outer_product(dpos)

            self.F[p] = (ti.Matrix.identity(self.real, self.dim) + self.dt * new_C) @ self.old_F[p]
            self.updateIsotropicHelper(p, self.F[p])
            self.scratch_xp[p] = self.x[p] + self.dt * self.scratch_vp[p]

    @ti.func
    def computeDvAndGradDv(self, dv : ti.template()):
        for p in self.x:
            Xp = self.x[p] * self.inv_dx
            base = int(Xp - 0.5)
            fx = Xp - base
            w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1)**2, 0.5 * (fx - 0.5)**2]
            vp = ti.zero(self.scratch_vp[p])
            gradV = ti.zero(self.scratch_gradV[p])

            for offset in ti.static(ti.grouped(ti.ndrange(*self.neighbour))):
                dpos = (offset - fx) * self.dx
                weight = self.real(1)
                for i in ti.static(range(self.dim)):
                    weight *= w[offset[i]][i]

                vp += weight * dv[p]
                gradV += 4 * self.inv_dx * weight * dv[p].outer_product(dpos)

            self.scratch_vp[p] = vp
            self.scratch_gradV[p] = gradV
        
    @ti.func
    def computeStressDifferential(self, p, gradDv : ti.template(), dstress : ti.template(), dvp : ti.template()):
        Fn_local = self.old_F[p]
        dP = ti.Matrix.zero(self.real, self.dim, self.dim)
        self.firstPiolaDifferential(p, Fn_local, gradDv @ Fn_local, dP)
        dstress += self.p_vol * Fn_local.determinant() * dP @ Fn_local.transpose()

    @ti.kernel
    def multiply(self, x : ti.template(), b : ti.template()):
        for I in b:
            b[I] = ti.zero(b[I])
        
        # Note the relationship H dx = - df, where H is the stiffness matrix
        # inertia part
        for I in x:
            b[I] += self.mass_matrix[I] * x[I]

        self.computeDvAndGradDv(x)

        # scratch_gradV is now temporaraly used for storing gradDV (evaluated at particles)
        # scratch_vp is now temporaraly used for storing DV (evaluated at particles)

        for p in self.x:
            self.scratch_stress[p] = ti.zero(self.scratch_stress[p])
        
        for p in self.x:
            self.computeStressDifferential(p, self.scratch_gradV[p], self.scratch_stress[p], self.scratch_vp[p])
            # scratch_stress is now V_p^0 dP (F_p^n)^T (dP is Ap in snow paper)

        ti.block_dim(self.n_grid)
        for p in self.x:
            Xp = self.x[p] * self.inv_dx
            base = int(Xp - 0.5)
            fx = Xp - base
            w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1)**2, 0.5 * (fx - 0.5)**2] # Quadratic kernels  [http://mpm.graphics   Eqn. 123, with x=fx, fx-1,fx-2]            
            stress = self.scratch_stress[p]
            
            for offset in ti.static(ti.grouped(ti.ndrange(*self.neighbour))):
                dpos = (offset - fx) * self.dx
                weight = self.real(1)
                for i in ti.static(range(self.dim)):
                    weight *= w[offset[i]][i]
                
                b[self.idx(base + offset)] -= self.dt * self.dt * (weight * stress @ dpos)

    @ti.func
    def project(self, x : ti.template()):
        for p in x:
            I = self.node(p)
            cond = any(I < self.bound) or any(I > self.n_grid - self.bound)
            if cond: x[p] = ti.zero(x[p])

    @ti.kernel
    def kernelProject(self, x : ti.template()):
        self.project(x)

    @ti.kernel
    def precondition(self, _in : ti.template(), _out : ti.template()):
        for I in _in:
            _out[I] = _in[I] / self.mass_matrix[I] if self.mass_matrix[I] > 0 else _in[I]

    @ti.kernel
    def dotProduct(self, a : ti.template(), b : ti.template()) -> ti.f32:
        result = self.real(0)
        for I in a:
            result += a[I].dot(b[I])

        return result
    
    @ti.kernel
    def linearSolverReinitialize(self):
        for I in self.mass_matrix:
            self.r[I] = ti.zero(self.r[I])
            self.p[I] = ti.zero(self.p[I])
            self.q[I] = ti.zero(self.q[I])
            self.temp[I] = ti.zero(self.temp[I])

    # solve Ax = b, where A build implicitly, x := step_direction, b := redisual
    def linearSolve(self, x, b, relative_tolerance):
        self.linear_solver_data.deactivate_all()
        self.linearSolverReinitialize()

        # NOTE: requires that the input x has been projected
        self.multiply(x, self.temp)
        self.kernelProject(self.r)
        self.precondition(self.r, self.q) # NOTE: requires that preconditioning matrix is projected
        self.copy(self.p, self.q)

        zTrk = self.dotProduct(self.r, self.q)
        residual_preconditioned_norm = ti.sqrt(zTrk)
        local_tolerance = self.real(ti.min(relative_tolerance * residual_preconditioned_norm, self.linear_solve_tolerance))
        for cnt in range(self.linear_solve_max_iterations):
            if residual_preconditioned_norm <= local_tolerance:
                return cnt

            self.multiply(self.p, self.temp)
            self.kernelProject(self.temp)
            alpha = zTrk / self.dotProduct(self.temp, self.p)

            self.copy(x, x, alpha, self.p) # i.e. x += p * alpha
            self.copy(self.r, self.r, -alpha, self.temp) # i.e. r -= temp * alpha
            self.precondition(self.r, self.q) # NOTE: requires that preconditioning matrix is projected

            zTrk_last = zTrk
            zTrk = self.dotProduct(self.q, self.r)
            beta = zTrk / zTrk_last

            self.copy(p, q, beta, p) # i.e. p = q + beta * p

            residual_preconditioned_norm = ti.sqrt(zTrk)

        return self.linear_solve_max_iterations

        
    def backwardEulerStep(self): # on the assumption that collision is ignored
        self.buildMassMatrix()
        self.buildInitialDvAndVnForNewton()
        # Which should be called at the beginning of newton.
        self.backupStrain()

        self.newtonSolve()

        self.restoreStrain()
        self.constructNewVelocityFromNewtonResult()

    def newtonSolve(self):
        self.updateState()
        E0 = 0 # totalEnergy of last iteration, for line search only
        if ti.static(self.line_search):
            E0 = self.totalEnergy()
            self.copy(self.dv0, self.dv)

        for it in range(self.newton_max_iterations):
            # Mv^(n) - Mv^(n+1) + dt * f(x_n + dt v^(n+1)) + dt * Mg
            # -Mdv + dt * f(x_n + dt(v^n + dv)) + dt * Mg
            self.computeResidual()
            residual_norm = self.computeNorm()
            if residual_norm < self.newton_tolerance:
                break

            linear_solve_relative_tolerance = ti.min(0.5, self.linear_solve_tolerance_scale * ti.sqrt(ti.max(residual_norm, self.newton_tolerance)))
            self.linearSolve(self.step_direction, self.residual, linear_solve_relative_tolerance)
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
            self.updateIsotropicHelper(p, self.F[p])
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

    gui = ti.GUI("Taichi IMPLICIT MPM", res = 512, background_color = 0x112F41)
    while gui.running:
        for i in range(10):
            solver.reinitialize()
            solver.particlesToGrid()
            # solver.gridUpdateExplicit()
            solver.backwardEulerStep()
            solver.gridToParticles()

        pos = solver.x.to_numpy()
        gui.circles(pos, radius=1.5, color=0x66ccff)
        gui.show()