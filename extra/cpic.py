import taichi as ti
import numpy as np

ti.init(arch = ti.gpu)

ID = ti.field(dtype = float, shape = ())
@ti.data_oriented
class RigidBody:
    def __init__(self):
        self.max_segments = 128
        self.num_segments = 0
        self.id = 0
        self.mesh = ti.Vector.field(2, dtype = float, shape = (self.max_segments, 2)) # 0 -> begin, 1 -> end

        self.centroid = ti.Vector.field(2, dtype = float, shape = ())
        self.angle = ti.field(dtype = float, shape = ())

        self.mass = ti.field(dtype = float, shape = ())
        self.I = ti.field(dtype = float, shape = ())

        self.velocity = ti.Vector.field(2, dtype = float, shape = ())
        self.angular = ti.field(dtype = float, shape = ())

    @ti.kernel
    def init(self, N, M, seg : ti.template()):
        self.id = ID[None]
        ID[None] += 1

        self.num_segments = N
        for i, j in seg:
            self.mesh[i, j] = seg[i, j]

        self.mass = M
        tot_length = 0.0
        for i in range(self.num_segments):
            length = (self.mesh[i, 1] - self.mesh[i, 0]).norm()
            tot_length += length

    @ti.func
    def update_p(self, dt):
        self.centroid[None] += dt * self.velocity
        self.angle += dt * self.angular

    @ti.func
    def update_v(self, px, py, fx, fy, dt):
        self.velocity += ti.Vector([fx, fy]) / self.mass * dt
        self.angular += ti.cross(ti.Vector([px - self.centroid.x, py - self.centroid.y]), ti.Vector([fx, fy])) / self.I * dt

