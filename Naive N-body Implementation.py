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
#---2. Force Calculation (Naive O(N^2))--
def calculate_forces(bodies, G, softening):
# Reset accelerations
for body in bodies:
body.reset_acceleration()
# Nested loop
for i in range(len(bodies)):
for j in range(len(bodies)):
if i == j: continue
body1 = bodies[i]
body2 = bodies[j]
dx = body2.x- body1.x
dy = body2.y- body1.y
dz = body2.z- body1.z
dist_sq = dx*dx + dy*dy + dz*dz
# Softening prevents division by zero
dist_soft = (dist_sq + softening**2)**(1.5)
# F = G * m1 * m2 / r^2-> a = F / m1
acc_factor = (G * body2.mass) / dist_soft
body1.ax += acc_factor * dx
body1.ay += acc_factor * dy
body1.az += acc_factor * dz
#--- 3. Leapfrog Integration--
def leapfrog_step(bodies, dt, G, softening):
# First Half-Kick
for body in bodies:
body.vx += 0.5 * body.ax * dt
body.vy += 0.5 * body.ay * dt
body.vz += 0.5 * body.az * dt
# Drift
for body in bodies:
body.x += body.vx * dt
body.y += body.vy * dt
body.z += body.vz * dt
# Update Forces
calculate_forces(bodies, G, softening)
# Second Half-Kick
for body in bodies:
body.vx += 0.5 * body.ax * dt
body.vy += 0.5 * body.ay * dt
body.vz += 0.5 * body.az * dt