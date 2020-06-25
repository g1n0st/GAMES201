# author: g1n0st
# Symplectic Euler Method for Mass-Spring System
import taichi as ti

ti.init(debug = True)

max_particles = 256                         # the maximum number of particles
dt = 1e-3                                   # iteration time step

bottom_y = 0.05                             # lower boundary y coordinate
particle_mass = 1                           # mass of single particle
connection_radius = 0.15                    # auto connect particles within radius distance
gravity = [0, -9.8]                         # gravity vector

num_particles = ti.var(ti.i32, shape = ())
damping = ti.var(ti.f32, shape = ())
stiffness = ti.var(ti.f32, shape = ())

x = ti.Vector(2, dt = ti.f32, shape = max_particles)
v = ti.Vector(2, dt = ti.f32, shape = max_particles)

# rest_length[i, j] = 0 means i and j are not connected
rest_length = ti.var(ti.f32, shape = (max_particles, max_particles))

# simulation paused and enter edit mode
paused = False

@ti.kernel
def new_particle(pos_x : ti.f32, pos_y : ti.f32):
    new_id = num_particles[None]
    num_particles += 1
    x[new_id] = [pos_x, pos_y]
    v[new_id] = [0, 0]

    # connect with existing particles
    for i in range(new_id):
        dist = (x[new_id] - x[i]).norm()
        if dist < connection_radius:
            rest_length[i, new_id] = dist
            rest_length[new_id, i] = dist

@ti.kernel
def edge_modify(pos_x : ti.f32, pos_y : ti.f32) :
    eps = 0.01

    for i in range(num_particles[None]):
        for j in range(i):
            A = x[j].y - x[i].y
            B = x[i].x - x[j].x
            C = x[j].x * x[i].y - x[i].x * x[j].y
            norm = ti.sqrt(A * A + B * B)
            dist = ti.abs((A * pos_x + B * pos_y + C) / norm)
            if dist < eps:
                if rest_length[i, j] != 0:
                    rest_length[i, j] = 0
                    rest_length[j, i] = 0
                else:
                    rest_length[i, j] = norm
                    rest_length[j, i] = norm
                return

# initial parameter status
stiffness[None] = 10000
damping[None] = 20

gui = ti.GUI('Mass-Spring System', res = (512, 512), background_color = 0xdddddd)

new_particle(0.3, 0.3)
new_particle(0.3, 0.4)
new_particle(0.4, 0.4)

while True:
    for e in gui.get_events(ti.GUI.PRESS):
        if e.key in [ti.GUI.ESCAPE, ti.GUI.EXIT]:
            exit()
        elif e.key == gui.SPACE:
            paused = not paused
        elif e.key == ti.GUI.LMB:
            new_particle(e.pos[0], e.pos[1])
        elif e.key == ti.GUI.RMB and paused:
            edge_modify(e.pos[0], e.pos[1])
        elif e.key == 'c':
            num_particles[None] = 0
            rest_length.fill(0)
        elif e.key == 's':
            if gui.is_pressed('Shift'):
                stiffness[None] /= 1.1
            else:
                stiffness[None] *= 1.1
        elif e.key == 'd':
            if gui.is_pressed('Shift'):
                damping[None] /= 1.1
            else:
                damping[None] *= 1.1

    X = x.to_numpy()
    gui.circles(X[:num_particles[None]], color = 0xffaa77, radius = 5)

    gui.line(begin = (0.0, bottom_y), end = (1.0, bottom_y), color = 0x0, radius = 1)

    for i in range(num_particles[None]):
        for j in range(i + 1, num_particles[None]):
            if rest_length[i, j] != 0:
                gui.line(begin = X[i], end = X[j], radius = 2, color = 0x445566)
    gui.show()
