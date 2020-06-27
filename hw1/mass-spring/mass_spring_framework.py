# author: g1n0st
# Spring particle system simulation framework,
# including UI interface, user input and editing operations
import taichi as ti
import numpy as np

ti.init(debug = False)

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
paused = ti.var(ti.i32, shape = ())

# collide with ground
@ti.func
def collide_with_ground():
    for i in range(num_particles[None]):
        if x[i].y < bottom_y:
            x[i].y = bottom_y
            v[i].y = 0

# compute new position
@ti.func
def update_position():
    for i in range(num_particles[None]):
        x[i] += v[i] * dt
        
@ti.kernel
def new_particle(pos_x : ti.f32, pos_y : ti.f32, mode : ti.i32):
    new_id = num_particles[None]
    num_particles[None] += 1
    x[new_id] = [pos_x, pos_y]
    v[new_id] = [0, 0]

    # ('n' Mode) Add particle without connection
    # ...
    # (Shift Mode) Add particle connect to last added particle
    if mode == 1 and new_id > 0:
        dist = (x[new_id] - x[new_id - 1]).norm()
        rest_length[new_id - 1, new_id] = dist
        rest_length[new_id, new_id - 1] = dist
        
    # (Normal Mode) connect with existing particles
    if mode == 2:
        for i in range(new_id):
            dist = (x[new_id] - x[i]).norm()
            if dist < connection_radius:
                rest_length[i, new_id] = dist
                rest_length[new_id, i] = dist

@ti.kernel
def modify_springs(pos_x : ti.f32, pos_y : ti.f32) :
    eps = 0.003

    for i in range(num_particles[None]):
        for j in range(i):
            # claculate the parameters of straight line general equation
            A = x[j].y - x[i].y
            B = x[i].x - x[j].x
            C = x[j].x * x[i].y - x[i].x * x[j].y
            norm = ti.sqrt(A * A + B * B)

            # calculate the distance of the point to the straight line
            dist = ti.abs((A * pos_x + B * pos_y + C) / norm)
            # r means AP.AB / ||AB||
            r = ((pos_y - x[i].y) * A - (pos_x - x[i].x) * B) / (norm * norm)
            if dist < eps and 0 < r < 1:
                if rest_length[i, j] != 0:
                    rest_length[i, j] = 0
                    rest_length[j, i] = 0
                else:
                    rest_length[i, j] = norm
                    rest_length[j, i] = norm
                break

cursor_pos = ti.Vector(2, dt = ti.f32, shape = ())
@ti.kernel
def modify_particles(pos_x : ti.f32, pos_y : ti.f32):
    eps = 0.015
    cursor_pos[None].x = pos_x
    cursor_pos[None].y = pos_y
    del_particle = -1
    n = num_particles[None]
    for i in range(n):
        if (cursor_pos[None] - x[i]).norm() < eps:
            del_particle = i

    if del_particle != -1:
        # remove i-th particle means use the last particle's data to replace it
        for i in range(n):
            rest_length[i, del_particle] = rest_length[i, n - 1]
            rest_length[del_particle, i] = rest_length[n - 1, i]
            rest_length[i, n - 1] = 0
            rest_length[n - 1, i] = 0

        x[del_particle] = x[n - 1]
        v[del_particle] = v[n - 1]
        x[n - 1] = [0, 0]
        v[n - 1] = [0, 0]
        num_particles[None] -= 1

# (green <-- black --> red)
# red means the spring is elongating
# green means the spring is compressing
@ti.kernel
def calculate_color(delta: ti.f32) -> ti.i32:
    sigmoid = 2 / (1 + ti.exp(-delta * stiffness[None] * 0.1)) - 1
    return int(max(sigmoid, 0) * 0xff) * 0x10000 - int(min(sigmoid, 0) * 0xff) * 0x100

gui = ti.GUI('Mass-Spring System', res = (512, 512), background_color = 0xdddddd)
def init_mass_spring_system():
# initial parameter status
    stiffness[None] = 10000
    damping[None] = 5
    paused[None] = False

    new_particle(0.3, 0.3, 2)
    new_particle(0.3, 0.4, 2)
    new_particle(0.4, 0.4, 2)

def process_input():
    for e in gui.get_events(ti.GUI.PRESS):
        if e.key in [ti.GUI.ESCAPE, ti.GUI.EXIT]:
            exit()
        elif e.key == gui.SPACE:
            paused[None] = not paused[None]

        # add particles in arbitrary state
        elif e.key == ti.GUI.LMB:
            if gui.is_pressed('Shift'):
                new_particle(e.pos[0], e.pos[1], 1)
            else:
                new_particle(e.pos[0], e.pos[1], 2)
        elif e.key == 'n':
            new_particle(e.pos[0], e.pos[1], 0)

        # edit particles and springs in pause state
        elif e.key == ti.GUI.RMB and paused[None]:
            modify_particles(e.pos[0], e.pos[1])
            modify_springs(e.pos[0], e.pos[1])

        # clear all particles and springs
        elif e.key == 'c':
            num_particles[None] = 0
            rest_length.fill(0)

        elif e.key == 's':
            if gui.is_pressed('Shift'):
                stiffness[None] /= 1.14514
            else:
                stiffness[None] *= 1.14514
        elif e.key == 'd':
            if gui.is_pressed('Shift'):
                damping[None] /= 1.14514
            else:
                damping[None] *= 1.14514

def process_output():
    X = x.to_numpy()
    gui.circles(X[:num_particles[None]], color = 0xffaa77, radius = 5)

    gui.line(begin = (0.0, bottom_y), end = (1.0, bottom_y), color = 0x0, radius = 2)

    for i in range(num_particles[None]):
        for j in range(i + 1, num_particles[None]):
            if rest_length[i, j] != 0:
                norm = np.linalg.norm(X[i] - X[j])
                gui.line(begin = X[i], end = X[j], radius = 2, color = calculate_color(norm - rest_length[i, j]))

    gui.text(content = f'C: clear all; Space: pause', pos = (0, 1), color = 0x0)
    gui.text(content = f'S: Spring stiffness {stiffness[None]:.1f}', pos = (0, 0.95), color = 0x0)
    gui.text(content = f'D: Damping {damping[None]:.2f}', pos = (0, 0.9), color = 0x0)
    if paused[None]:
        gui.text(content = f'Edit Mode', pos = (0, 0.85), color = 0x0)
    gui.show()
