import sys
import math
import numpy as np
import pandas as pd
import time

# ---------------------------------------------------------
# Constants
# ---------------------------------------------------------
G = 6.67430e-11
softening = 1e9
theta = 0.5
dt = 8640.0
steps = 100

max_nodes = None

# Node pool arrays (Will be initialized as NumPy arrays)
node_cx = node_cy = node_cz = node_size = node_mass = None
node_com_x = node_com_y = node_com_z = None
node_child = node_particle = node_is_leaf = None
node_xmin = node_xmax = node_ymin = node_ymax = node_zmin = node_zmax = None

node_count = 0

# Body arrays (NumPy arrays)
n = 0
mass = px = py = pz = vx = vy = vz = ax = ay = az = None

# ---------------------------------------------------------
# 1. Exact M3 Tree Building Logic
# ---------------------------------------------------------
def allocate_node():
    global node_count
    idx = node_count
    node_count += 1
    
    node_mass[idx] = 0.0
    node_com_x[idx] = 0.0
    node_com_y[idx] = 0.0
    node_com_z[idx] = 0.0
    node_particle[idx] = -1
    node_is_leaf[idx] = 1
    
    return idx

def get_octant(node_idx, px_val, py_val, pz_val):
    cx = node_cx[node_idx]
    cy = node_cy[node_idx]
    cz = node_cz[node_idx]
    
    oct = 0
    if px_val > cx: oct |= 4
    if py_val > cy: oct |= 2
    if pz_val > cz: oct |= 1
    return oct

def subdivide(node_idx):
    global node_count
    
    xmin, xmax = node_xmin[node_idx], node_xmax[node_idx]
    ymin, ymax = node_ymin[node_idx], node_ymax[node_idx]
    zmin, zmax = node_zmin[node_idx], node_zmax[node_idx]
    cx, cy, cz = node_cx[node_idx], node_cy[node_idx], node_cz[node_idx]
    
    half = node_size[node_idx] / 2.0
    quarter = half / 2.0
    
    child_centers = []
    for dz in (-quarter, quarter):
        for dy in (-quarter, quarter):
            for dx in (-quarter, quarter):
                child_centers.append((cx + dx, cy + dy, cz + dz))
    
    for octant, (ccx, ccy, ccz) in enumerate(child_centers):
        child_idx = allocate_node()
        
        node_cx[child_idx] = ccx
        node_cy[child_idx] = ccy
        node_cz[child_idx] = ccz
        node_size[child_idx] = half
        
        child_half = half / 2.0
        node_xmin[child_idx] = ccx - child_half
        node_xmax[child_idx] = ccx + child_half
        node_ymin[child_idx] = ccy - child_half
        node_ymax[child_idx] = ccy + child_half
        node_zmin[child_idx] = ccz - child_half
        node_zmax[child_idx] = ccz + child_half
        
        node_child[node_idx * 8 + octant] = child_idx
    
    node_is_leaf[node_idx] = 0
    node_particle[node_idx] = -1

def insert_particle(body_idx, node_idx):
    if node_is_leaf[node_idx]:
        if node_particle[node_idx] == -1:
            node_particle[node_idx] = body_idx
            node_mass[node_idx] = mass[body_idx]
            node_com_x[node_idx] = px[body_idx]
            node_com_y[node_idx] = py[body_idx]
            node_com_z[node_idx] = pz[body_idx]
            return
        else:
            subdivide(node_idx)
            existing = node_particle[node_idx]
            insert_particle(existing, node_idx)
            insert_particle(body_idx, node_idx)
            return
    else:
        oct = get_octant(node_idx, px[body_idx], py[body_idx], pz[body_idx])
        child_idx = node_child[node_idx * 8 + oct]
        if child_idx == -1:
            raise RuntimeError("child missing in internal node")
        insert_particle(body_idx, child_idx)

def compute_mass(node_idx):
    if node_is_leaf[node_idx]:
        if node_particle[node_idx] != -1:
            b = node_particle[node_idx]
            node_mass[node_idx] = mass[b]
            node_com_x[node_idx] = px[b]
            node_com_y[node_idx] = py[b]
            node_com_z[node_idx] = pz[b]
        else:
            node_mass[node_idx] = 0.0
        return
    
    total_mass = sum_cx = sum_cy = sum_cz = 0.0
    
    for k in range(8):
        child = node_child[node_idx * 8 + k]
        if child != -1:
            compute_mass(child)
            m = node_mass[child]
            if m > 0:
                total_mass += m
                sum_cx += node_com_x[child] * m
                sum_cy += node_com_y[child] * m
                sum_cz += node_com_z[child] * m
    
    node_mass[node_idx] = total_mass
    if total_mass > 0:
        node_com_x[node_idx] = sum_cx / total_mass
        node_com_y[node_idx] = sum_cy / total_mass
        node_com_z[node_idx] = sum_cz / total_mass
    else:
        node_com_x[node_idx] = 0.0
        node_com_y[node_idx] = 0.0
        node_com_z[node_idx] = 0.0

def compute_bounding_box():
    min_x, max_x = np.min(px), np.max(px)
    min_y, max_y = np.min(py), np.max(py)
    min_z, max_z = np.min(pz), np.max(pz)
    
    cx = (min_x + max_x) / 2.0
    cy = (min_y + max_y) / 2.0
    cz = (min_z + max_z) / 2.0
    
    max_width = max(max_x - min_x, max_y - min_y, max_z - min_z)
    padding = max_width * 0.01
    size = max_width + 2 * padding
    
    if size <= 0: size = 1e10
    return cx, cy, cz, size

def build_tree():
    global node_count
    node_count = 0
    
    # Fast vectorized reset for NumPy arrays
    node_mass.fill(0.0)
    node_com_x.fill(0.0)
    node_com_y.fill(0.0)
    node_com_z.fill(0.0)
    node_particle.fill(-1)
    node_is_leaf.fill(1)
    node_child.fill(-1)
    
    cx, cy, cz, size = compute_bounding_box()
    root = allocate_node()
    
    node_cx[root] = cx
    node_cy[root] = cy
    node_cz[root] = cz
    node_size[root] = size
    
    half = size / 2.0
    node_xmin[root] = cx - half
    node_xmax[root] = cx + half
    node_ymin[root] = cy - half
    node_ymax[root] = cy + half
    node_zmin[root] = cz - half
    node_zmax[root] = cz + half
    
    for i in range(n):
        insert_particle(i, root)
    
    compute_mass(root)
    return root

# ---------------------------------------------------------
# 2. Vectorized NumPy Force Calculation (Milestone 4)
# ---------------------------------------------------------
def build_interaction_list(body_idx, root_idx):
    interaction_list = []
    stack = [root_idx]
    
    while stack:
        node = stack.pop()
        
        if node_mass[node] == 0.0:
            continue
            
        if node_is_leaf[node] and node_particle[node] == body_idx:
            continue
            
        dx = node_com_x[node] - px[body_idx]
        dy = node_com_y[node] - py[body_idx]
        dz = node_com_z[node] - pz[body_idx]
        r = math.sqrt(dx*dx + dy*dy + dz*dz)
        
        if node_is_leaf[node] or (node_size[node] / r < theta):
            interaction_list.append(node)
        else:
            for k in range(8):
                child = node_child[node * 8 + k]
                if child != -1:
                    stack.append(child)
                    
    return interaction_list

def compute_force_numpy(i, interaction_list):
    if not interaction_list:
        return 0.0, 0.0, 0.0
        
    # Convert list to array for vectorized gather indexing
    nodes = np.array(interaction_list, dtype=np.int32)
    
    # Vectorized Math Ops (Equivalent to RVV Gather & Math)
    dx = node_com_x[nodes] - px[i]
    dy = node_com_y[nodes] - py[i]
    dz = node_com_z[nodes] - pz[i]
    
    dist_sq = dx*dx + dy*dy + dz*dz + (softening**2)
    inv_dist3 = dist_sq ** -1.5
    
    f = G * node_mass[nodes] * inv_dist3 
    
    return np.sum(f * dx), np.sum(f * dy), np.sum(f * dz)

def calculate_forces_bh_numpy(root):
    ax.fill(0.0)
    ay.fill(0.0)
    az.fill(0.0)
    
    for i in range(n):
        interaction_list = build_interaction_list(i, root)
        a_x, a_y, a_z = compute_force_numpy(i, interaction_list)
        ax[i] = a_x
        ay[i] = a_y
        az[i] = a_z

# ---------------------------------------------------------
# 3. Integration & Execution
# ---------------------------------------------------------
def kick_half_step(dt):
    global vx, vy, vz
    vx += 0.5 * ax * dt
    vy += 0.5 * ay * dt
    vz += 0.5 * az * dt

def drift(dt):
    global px, py, pz
    px += vx * dt
    py += vy * dt
    pz += vz * dt

def load_csv(filename):
    global n, mass, px, py, pz, vx, vy, vz, ax, ay, az, max_nodes
    global node_cx, node_cy, node_cz, node_size, node_mass
    global node_com_x, node_com_y, node_com_z, node_child
    global node_particle, node_is_leaf, node_xmin, node_xmax
    global node_ymin, node_ymax, node_zmin, node_zmax
    
    df = pd.read_csv(filename)
    n = len(df)
    
    # Load body parameters directly into NumPy arrays
    mass = df['mass'].values.astype(np.float64)
    px = df['distanceX'].values.astype(np.float64)
    py = df['distanceY'].values.astype(np.float64)
    pz = df['distanceZ'].values.astype(np.float64)
    vx = df['velocityX'].values.astype(np.float64)
    vy = df['velocityY'].values.astype(np.float64)
    vz = df['velocityZ'].values.astype(np.float64)
    
    ax = np.zeros(n, dtype=np.float64)
    ay = np.zeros(n, dtype=np.float64)
    az = np.zeros(n, dtype=np.float64)
    
    # Initialize Node Pool as NumPy arrays
    max_nodes = 8 * n + 100
    
    node_cx = np.zeros(max_nodes, dtype=np.float64)
    node_cy = np.zeros(max_nodes, dtype=np.float64)
    node_cz = np.zeros(max_nodes, dtype=np.float64)
    node_size = np.zeros(max_nodes, dtype=np.float64)
    node_mass = np.zeros(max_nodes, dtype=np.float64)
    node_com_x = np.zeros(max_nodes, dtype=np.float64)
    node_com_y = np.zeros(max_nodes, dtype=np.float64)
    node_com_z = np.zeros(max_nodes, dtype=np.float64)
    
    node_child = np.full(max_nodes * 8, -1, dtype=np.int32)
    node_particle = np.full(max_nodes, -1, dtype=np.int32)
    node_is_leaf = np.ones(max_nodes, dtype=np.int32)
    
    node_xmin = np.zeros(max_nodes, dtype=np.float64)
    node_xmax = np.zeros(max_nodes, dtype=np.float64)
    node_ymin = np.zeros(max_nodes, dtype=np.float64)
    node_ymax = np.zeros(max_nodes, dtype=np.float64)
    node_zmin = np.zeros(max_nodes, dtype=np.float64)
    node_zmax = np.zeros(max_nodes, dtype=np.float64)

def run_simulation(csv_filename, dt=dt, steps=steps):
    load_csv(csv_filename)
    print(f"Loaded {n} bodies from {csv_filename}")
    
    # Initial setup
    root = build_tree()
    calculate_forces_bh_numpy(root)
    
    print("Starting NumPy-accelerated time integration...")
    
    start_time = time.perf_counter()
    
    for step in range(1, steps + 1):
        kick_half_step(dt)
        drift(dt)
        root = build_tree()
        calculate_forces_bh_numpy(root)
        kick_half_step(dt)
        
    end_time = time.perf_counter()
    print(f"Completed {steps} steps in {end_time - start_time:.4f} seconds.")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        run_simulation(sys.argv[1])
    else:
        run_simulation("stable_random_system100.csv")