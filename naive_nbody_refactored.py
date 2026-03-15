import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# naive implementation refactored code using arrays 

# funct to load all bodies from csv file and convert to arrays
def load_bodies_from_csv(filename):
    df = pd.read_csv(filename)
    N = len(df)

    # all the body structs are flattened to arrays 
    mass = [0.0] * N
    x = [0.0] * N
    y = [0.0] * N
    z = [0.0] * N
    vx = [0.0] * N
    vy = [0.0] * N
    vz = [0.0] * N
    ax = [0.0] * N
    ay = [0.0] * N
    az = [0.0] * N

    for i in range(N):
        mass[i] = df.loc[i, 'mass']
        x[i] = df.loc[i, 'distanceX']
        y[i] = df.loc[i, 'distanceY']
        z[i] = df.loc[i, 'distanceZ']
        vx[i] = df.loc[i, 'velocityX']
        vy[i] = df.loc[i, 'velocityY']
        vz[i] = df.loc[i, 'velocityZ']

    return N, mass, x, y, z, vx, vy, vz, ax, ay, az


# func to reset accerlerations so that it doesnt use the old accer to calculate the new one 
def reset_accelerations(N, ax, ay, az):
    for i in range(N):
        ax[i] = 0.0
        ay[i] = 0.0
        az[i] = 0.0


# fun to calculate forces 
def calculate_forces(N, mass, x, y, z, ax, ay, az, G, softening):
    reset_accelerations(N, ax, ay, az)

    for i in range(N):
        for j in range(N):
            if i == j:
                continue

            dx = x[j] - x[i]
            dy = y[j] - y[i]
            dz = z[j] - z[i]

            dist_sq = dx * dx + dy * dy + dz * dz
            dist_soft = (dist_sq + softening ** 2) ** 1.5 

            acc_factor = (G * mass[j]) / dist_soft
            ax[i] += acc_factor * dx
            ay[i] += acc_factor * dy
            az[i] += acc_factor * dz


# func leapfrong integration steps
def leapfrog_step(N, mass, x, y, z, vx, vy, vz, ax, ay, az, dt, G, softening):
    # 1st half kick
    for i in range(N):
        vx[i] += 0.5 * ax[i] * dt
        vy[i] += 0.5 * ay[i] * dt
        vz[i] += 0.5 * az[i] * dt

    # drift
    for i in range(N):
        x[i] += vx[i] * dt
        y[i] += vy[i] * dt
        z[i] += vz[i] * dt

    # update the forces 
    calculate_forces(N, mass, x, y, z, ax, ay, az, G, softening)

    # 2nd half kick
    for i in range(N):
        vx[i] += 0.5 * ax[i] * dt
        vy[i] += 0.5 * ay[i] * dt
        vz[i] += 0.5 * az[i] * dt


# func for simulationa and visualisation 
def run_simulation():
    # gravitational const
    G = 6.67430e-11
    # softening length 1000000 km
    softening = 1e9
    # 1 day per steps, so total no. of seconds which is 24 hours * 60 * 60
    dt = 864000
    # no of frames
    steps = 200

    filename = 'stable_random_system100.csv'
    N, mass, x, y, z, vx, vy, vz, ax, ay, az = load_bodies_from_csv(filename)
    print(f"Loaded {N} bodies from {filename}")

    calculate_forces(N, mass, x, y, z, ax, ay, az, G, softening)

    fig, ax_plot = plt.subplots(figsize=(8, 8))
    limit = 5e12
    ax_plot.set_xlim(-limit, limit)
    ax_plot.set_ylim(-limit, limit)
    ax_plot.set_aspect('equal')
    ax_plot.set_facecolor('black')

    points, = ax_plot.plot([], [], 'o', ms=2, color='white')

    def update(frame):
        leapfrog_step(N, mass, x, y, z, vx, vy, vz, ax, ay, az, dt, G, softening)
        x_data = [x[i] for i in range(N)]
        y_data = [y[i] for i in range(N)]
        points.set_data(x_data, y_data)
        return points,

    print("Running Simulation Window...")
    anim = FuncAnimation(fig, update, frames=steps, interval=20, blit=True)
    plt.show()


if __name__ == "__main__":
    run_simulation()