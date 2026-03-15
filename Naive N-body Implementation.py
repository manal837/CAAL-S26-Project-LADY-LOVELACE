import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

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


#---4. File Loader (New!)--
def load_bodies_from_csv(filename):
df = pd.read_csv(filename)
bodies = []

# Mapping CSV columns to our class
for index, row in df.iterrows():
b = Body(
mass=row[’mass’],
x=row[’distanceX’],
y=row[’distanceY’],
z=row[’distanceZ’],
vx=row[’velocityX’],
vy=row[’velocityY’],
vz=row[’velocityZ’]
)
bodies.append(b)
return bodies


#---5. Main Simulation & Visualization--
def run_simulation():
  
# REAL PHYSICS CONSTANTS
G = 6.67430e-11 # Real Gravitational Constant
softening = 1e9 # 1,000,000 km softening
dt = 3600 * 24 # 1 Day per step
steps = 200 # Number of frames

# Load Data (Change filename to 500 or 1000 to test others)
filename = ’stable_random_system100.csv’
bodies = load_bodies_from_csv(filename)
print(f"Loaded {len(bodies)} bodies from {filename}")

# Initial Force Calculation
calculate_forces(bodies, G, softening)

# Visualization Setup
fig, ax = plt.subplots(figsize=(8, 8))
limit = 5e12 # 5 trillion meters

ax.set_xlim(-limit, limit)
ax.set_ylim(-limit, limit)
ax.set_aspect(’equal’)
ax.set_facecolor(’black’)
points, = ax.plot([], [], ’o’, ms=2, color=’white’)

def update(frame):
leapfrog_step(bodies, dt, G, softening)
x_data = [b.x for b in bodies]
y_data = [b.y for b in bodies]
points.set_data(x_data, y_data)
return points,

print("Running Simulation Window...")
anim = FuncAnimation(fig, update, frames=steps, interval=20, blit=
True)
plt.show()

if __name__ == "__main__":
run_simulation()