#--- 1. The Body Class--
class Body:
def __init__(self, mass, x, y, z, vx, vy, vz):
self.mass = mass
self.x = x
self.y = y
self.z = z
self.vx = vx
self.vy = vy
self.vz = vz
self.ax = 0.0
self.ay = 0.0
self.az = 0.0
def reset_acceleration(self):
self.ax = 0.0
self.ay = 0.0
self.az = 0.0