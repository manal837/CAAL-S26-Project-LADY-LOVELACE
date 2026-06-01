# barnes-hut n-body simulation - refactored python version but for milestone 4 NUMPY 
# manal fatima 29018
import sys
import math
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# constants
G = 6.67430e-11
softening = 1e9  # softening length in metres
theta = 0.5      # opening angle for barnes hut
dt = 8640.0      # time step in seconds, match c++ example
steps = 100

# global node pool (sized after reading n)
# we set max_nodes = 8 * n + 100
max_nodes = None

# flat arrays for all node fields
node_cx = None      # center x of node
node_cy = None
node_cz = None
node_size = None    # side length of cube
node_mass = None
node_com_x = None
node_com_y = None
node_com_z = None
node_child = None   # flat: index node*8+oct gives child index, -1 = empty
node_particle = None # particle index if leaf, else -1
node_is_leaf = None  # 1 if leaf, 0 if internal
node_xmin = None
node_xmax = None
node_ymin = None
node_ymax = None
node_zmin = None
node_zmax = None

node_count = 0      # next free node index

# numpy node arrays, these will be built after the tree is constructed
np_node_com_x  = None
np_node_com_y  = None
np_node_com_z  = None
np_node_mass   = None
np_node_size   = None
np_node_child  = None
np_node_is_leaf= None
np_node_particle=None

# body arrays (global for simplicity, matches assembly layout)
n = 0
mass = None
px = None
py = None
pz = None
vx = None
vy = None
vz = None
ax = None
ay = None
az = None

# numpy body arrays for vectorised force cal
np_mass = None
np_px = None; 
np_py = None; 
np_pz = None
np_ax = None; 
np_ay = None; 
np_az = None

# func allocate_node which gets the next free node from pool
# returns: idx = new node index
# increments node_count and returns the previous value
def allocate_node():
    global node_count
    idx = node_count
    node_count += 1
    # init fields to default values
    node_mass[idx] = 0.0
    node_com_x[idx] = 0.0
    node_com_y[idx] = 0.0
    node_com_z[idx] = 0.0
    node_particle[idx] = -1
    node_is_leaf[idx] = 1
    # node_child already -1 from array init
    return idx


# func get_octant figures out which of 8 octants a point falls into
# uses bit encoding: bit2 (4) for x >= cx, bit1 (2) for y >= cy, bit0 (1) for z >= cz
# args: node_idx, px_val, py_val, pz_val
# returns: octant 0 to 7
def get_octant(node_idx, px_val, py_val, pz_val):
    # load node center
    cx = node_cx[node_idx]
    cy = node_cy[node_idx]
    cz = node_cz[node_idx]
    oct = 0
    if px_val > cx:
        oct |= 4          # set x bit (right half)
    if py_val > cy:
        oct |= 2          # set y bit (top half)
    if pz_val > cz:
        oct |= 1          # set z bit (front half)
    
    return oct


# func subdivide split a leaf node into 8 children (that is one per octant)
# args: node_idx - the node to subdivide (must be leaf)
# does: allocates 8 children, sets their centers, sizes, bounding boxes and marks parent as internal (is_leaf = 0)
def subdivide(node_idx):
    cx   = node_cx[node_idx]
    cy   = node_cy[node_idx]
    cz   = node_cz[node_idx]
    half = node_size[node_idx] / 2.0
    q    = half / 2.0

    for oct in range(8):
        dx = +q if (oct & 4) else -q
        dy = +q if (oct & 2) else -q
        dz = +q if (oct & 1) else -q

        child_idx = allocate_node()
        node_cx[child_idx]   = cx + dx
        node_cy[child_idx]   = cy + dy
        node_cz[child_idx]   = cz + dz
        node_size[child_idx] = half
        node_xmin[child_idx] = (cx + dx) - q
        node_xmax[child_idx] = (cx + dx) + q
        node_ymin[child_idx] = (cy + dy) - q
        node_ymax[child_idx] = (cy + dy) + q
        node_zmin[child_idx] = (cz + dz) - q
        node_zmax[child_idx] = (cz + dz) + q
        node_child[node_idx * 8 + oct] = child_idx

    node_is_leaf[node_idx]  = 0
    node_particle[node_idx] = -1


# func insert_particle inserts a body into the octree
# uses recursion: goes down tree until finds empty leaf if leaf occupied, subdivide then re-insert both bodies

def insert_particle(body_idx, node_idx):
    # case 1: node is leaf
    if node_is_leaf[node_idx]:
        # empty leaf - place body here
        if node_particle[node_idx] == -1:
            node_particle[node_idx] = body_idx
            node_mass[node_idx] = mass[body_idx]
            node_com_x[node_idx] = px[body_idx]
            node_com_y[node_idx] = py[body_idx]
            node_com_z[node_idx] = pz[body_idx]
            return
        else:
             # re-insert existing body that was here
            existing = node_particle[node_idx]  # old body index
            # occupied leaf - subdivide and re-insert both
            subdivide(node_idx)
            insert_particle(existing, node_idx)
            # insert new body
            insert_particle(body_idx, node_idx)
            return
    else:
        # case 2: internal node - find correct child and recurse
        oct = get_octant(node_idx, px[body_idx], py[body_idx], pz[body_idx])
        child_idx = node_child[node_idx * 8 + oct]
        # child should exist because subdivide creates all 8
        if child_idx == -1:
            raise RuntimeError("child missing in internal node")
        insert_particle(body_idx, child_idx)


# func compute_mass does post-order traversal to compute mass and com
# args: node_idx - current node
# does: for leaves, mass/com already set from insert for internal nodes, sum over children then divide

def compute_mass(node_idx):
    # leaf node - mass and com already set, nothing to do
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
    
    # internal node - sum over all 8 children
    total_mass = 0.0
    sum_cx = 0.0
    sum_cy = 0.0
    sum_cz = 0.0
    
    for k in range(8):
        child = node_child[node_idx * 8 + k]
        if child != -1:
            # recurse into child first (post-order)
            compute_mass(child)
            m = node_mass[child]
            if m > 0:
                total_mass += m
                sum_cx += node_com_x[child] * m
                sum_cy += node_com_y[child] * m
                sum_cz += node_com_z[child] * m
    
    # store total mass and weighted com
    node_mass[node_idx] = total_mass
    if total_mass > 0:
        node_com_x[node_idx] = sum_cx / total_mass
        node_com_y[node_idx] = sum_cy / total_mass
        node_com_z[node_idx] = sum_cz / total_mass
    else:
        node_com_x[node_idx] = 0.0
        node_com_y[node_idx] = 0.0
        node_com_z[node_idx] = 0.0


# calculate_force_bh - recursive force calc using barnes-hut criterion
# args: body_idx, node_idx, theta (opening angle)
# does: if node is far (size/dist < theta) treat as point mass else recurses into children

def calculate_force_bh(body_idx, node_idx, theta_val):
    if node_mass[node_idx] == 0.0:
        return
    if node_is_leaf[node_idx] and node_particle[node_idx] == body_idx:
        return   # skip self

    dx = node_com_x[node_idx] - px[body_idx]
    dy = node_com_y[node_idx] - py[body_idx]
    dz = node_com_z[node_idx] - pz[body_idx]
    r2 = dx*dx + dy*dy + dz*dz
    r  = math.sqrt(r2)

    if node_is_leaf[node_idx] or (r > 0.0 and node_size[node_idx] / r < theta_val):
        r_soft  = math.sqrt(r2 + softening * softening)
        r_cubed = r_soft * r_soft * r_soft
        factor  = G * node_mass[node_idx] / r_cubed
        ax[body_idx] += factor * dx
        ay[body_idx] += factor * dy
        az[body_idx] += factor * dz
    else:
        for k in range(8):
            child = node_child[node_idx * 8 + k]
            if child != -1:
                calculate_force_bh(body_idx, child, theta_val)

# func calculate_forces_bfs_numpy

def calculate_forces_bfs_numpy(root):
    global np_ax, np_ay, np_az
 
    # zero accelerations
    np_ax[:] = 0.0
    np_ay[:] = 0.0
    np_az[:] = 0.0
 
    # queue entries are (node_idx, particle_indices_array)
    # start: root node must be checked by all particles
    all_particles = np.arange(n, dtype=np.int32)
    queue = [(root, all_particles)]
 
    while len(queue) > 0:
        next_queue = []
 
        for node_idx, particle_indices in queue:
 
            # skip empty nodes
            if np_node_mass[node_idx] == 0.0:
                continue
 
            # skip if no particles to process
            if len(particle_indices) == 0:
                continue
 
            # get positions of all particles in this batch
            bx = np_px[particle_indices]
            by = np_py[particle_indices]
            bz = np_pz[particle_indices]
 
            # vector from each particle to node centre of mass
            dx = np_node_com_x[node_idx] - bx   # shape (k,)
            dy = np_node_com_y[node_idx] - by
            dz = np_node_com_z[node_idx] - bz
 
            # distance for all k particles at once
            r2 = dx*dx + dy*dy + dz*dz           # shape (k,)
            r  = np.sqrt(r2)                      # shape (k,)
 

            s = np_node_size[node_idx]
            criterion = (r > 0.0) & (s / (r + 1e-30) < theta)   # shape (k,) boolean
 
            # handle leaf nodes - must apply force directly (no children)
            is_leaf = np_node_is_leaf[node_idx] == 1
 
            if is_leaf:
                # leaf: apply force to all particles except self
                leaf_particle = np_node_particle[node_idx]
                if leaf_particle >= 0:
                    # remove self-interaction: mask out the particle that lives here
                    not_self = particle_indices != leaf_particle
                    apply_force_vectorized(node_idx, particle_indices[not_self])
                else:
                    apply_force_vectorized(node_idx, particle_indices)
                continue
 
            close_enough = particle_indices[criterion]
            if len(close_enough) > 0:
                apply_force_vectorized(node_idx, close_enough)
 
            # particles where criterion is False -> node too close -> open children
            need_to_open = particle_indices[~criterion]
            if len(need_to_open) > 0:
                for k in range(8):
                    child = int(np_node_child[node_idx * 8 + k])
                    if child != -1 and np_node_mass[child] > 0.0:
                        next_queue.append((child, need_to_open))
 
        queue = next_queue
 
    # copy results back to python lists for leapfrog integrator
    for i in range(n):
        ax[i] = float(np_ax[i])
        ay[i] = float(np_ay[i])
        az[i] = float(np_az[i])

# func compute_bounding_box finds global bounding box of all bodies
# returns: cx, cy, cz (center), size (cube side with 1% padding)
def compute_bounding_box():
    min_x = min(px[:n]);  max_x = max(px[:n])
    min_y = min(py[:n]);  max_y = max(py[:n])
    min_z = min(pz[:n]);  max_z = max(pz[:n])
    cx = (min_x + max_x) / 2.0
    cy = (min_y + max_y) / 2.0
    cz = (min_z + max_z) / 2.0
    max_width = max(max_x-min_x, max_y-min_y, max_z-min_z)
    size = max_width * 1.02
    if size <= 0.0:
        size = 1e10
    return cx, cy, cz, size


# build_tree - construct octree from current particle positions
# returns: root node index
# does: resets node pool, computes bbox, inserts all bodies, computes mass

def build_tree():
    global node_count, max_nodes
    
    # reset node pool
    node_count = 0
    
    # reset all node fields to defaults
    for i in range(max_nodes):
        node_mass[i] = 0.0
        node_com_x[i] = 0.0
        node_com_y[i] = 0.0
        node_com_z[i] = 0.0
        node_particle[i] = -1
        node_is_leaf[i] = 1
    
    for i in range(max_nodes * 8):
        node_child[i] = -1
    
    # compute bounding box and create the root node
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
    
    # insert all bodies
    for i in range(n):
        insert_particle(i, root)
    # compute masses and center of mass bottom-up
    compute_mass(root)
    return root


def build_numpy_arrays():
    global np_node_com_x, np_node_com_y, np_node_com_z
    global np_node_mass, np_node_size
    global np_node_child, np_node_is_leaf, np_node_particle
    global np_mass, np_px, np_py, np_pz
    global np_ax, np_ay, np_az
 
    nc = node_count   # only copy nodes that were actually allocated
 
    np_node_com_x   = np.array(node_com_x[:nc],   dtype=np.float64)
    np_node_com_y   = np.array(node_com_y[:nc],   dtype=np.float64)
    np_node_com_z   = np.array(node_com_z[:nc],   dtype=np.float64)
    np_node_mass    = np.array(node_mass[:nc],     dtype=np.float64)
    np_node_size    = np.array(node_size[:nc],     dtype=np.float64)
    np_node_child   = np.array(node_child[:nc*8],  dtype=np.int32)
    np_node_is_leaf = np.array(node_is_leaf[:nc],  dtype=np.int32)
    np_node_particle= np.array(node_particle[:nc], dtype=np.int32)
 
    np_mass = np.array(mass, dtype=np.float64)
    np_px   = np.array(px,   dtype=np.float64)
    np_py   = np.array(py,   dtype=np.float64)
    np_pz   = np.array(pz,   dtype=np.float64)
    np_ax   = np.zeros(n,    dtype=np.float64)
    np_ay   = np.zeros(n,    dtype=np.float64)
    np_az   = np.zeros(n,    dtype=np.float64)
 

# func apply_force_vectorized
def apply_force_vectorized(node_idx, particle_indices):
    # get positions of all particles in this batch
    # particle_indices is a numpy array e.g. [0, 3, 7, 12, ...]
    bx = np_px[particle_indices]   # shape (k,)
    by = np_py[particle_indices]
    bz = np_pz[particle_indices]
 

    dx = np_node_com_x[node_idx] - bx   # shape (k,)
    dy = np_node_com_y[node_idx] - by
    dz = np_node_com_z[node_idx] - bz
 
    # softened distance squared and cubed - all k particles at once
    r2      = dx*dx + dy*dy + dz*dz + softening*softening   # shape (k,)
    r_soft  = np.sqrt(r2)                                    # shape (k,)
    r_cubed = r2 * r_soft                                    # shape (k,)
 
    # gravitational factor for all k particles at once
    factor = G * np_node_mass[node_idx] / r_cubed           # shape (k,)
 
    # accumulate force contributions using numpy indexed assignment
    np_ax[particle_indices] += factor * dx
    np_ay[particle_indices] += factor * dy
    np_az[particle_indices] += factor * dz
 

# func calculate_forces_bh wrapper: zeros accelleration then compute all forces
def calculate_forces_scalar(root):
    # zero all accelerations
    for i in range(n):
        ax[i] = 0.0
        ay[i] = 0.0
        az[i] = 0.0
    
    # compute force on each body
    for i in range(n):
        calculate_force_bh(i, root, theta)


# func kick_half_step -leapfrog half-kick: v += a * dt/2
def kick_half_step(dt_val):
    for i in range(n):
        vx[i] += 0.5 * ax[i] * dt_val
        vy[i] += 0.5 * ay[i] * dt_val
        vz[i] += 0.5 * az[i] * dt_val


# func drift -leapfrog drift: p += v * dt
def drift(dt_val):
    for i in range(n):
        px[i] += vx[i] * dt_val
        py[i] += vy[i] * dt_val
        pz[i] += vz[i] * dt_val


# func calculate_energy - compute total energy (kinetic + potential)
# returns: total energy in joules
def calculate_energy():
    kin = 0.0
    pot = 0.0
    
    for i in range(n):
        v2   = vx[i]*vx[i] + vy[i]*vy[i] + vz[i]*vz[i]
        kin += 0.5 * mass[i] * v2
        
    for i in range(n):
        for j in range(i + 1, n):
            dx = px[i]-px[j];  
            dy = py[i]-py[j]; 
            dz = pz[i]-pz[j]
            r  = math.sqrt(dx*dx + dy*dy + dz*dz + softening*softening)
            pot -= G * mass[i] * mass[j] / r
    return kin + pot


# test - print simulation state for debugging
def test(step):
    total_mass = sum(mass)
    com_x = sum(mass[i]*px[i] for i in range(n)) / total_mass
    com_y = sum(mass[i]*py[i] for i in range(n)) / total_mass
    com_z = sum(mass[i]*pz[i] for i in range(n)) / total_mass
    mom_x = sum(mass[i]*vx[i] for i in range(n))
    mom_y = sum(mass[i]*vy[i] for i in range(n))
    mom_z = sum(mass[i]*vz[i] for i in range(n))
    energy = calculate_energy()
    print(f"step {step:4d}: E={energy:.6e}  com=({com_x:.3e},{com_y:.3e},{com_z:.3e})  mom=({mom_x:.3e},{mom_y:.3e},{mom_z:.3e})")


# func to load_csv which reads body data from csv file into global arrays
def load_csv(filename):
    global n, mass, px, py, pz, vx, vy, vz, ax, ay, az, max_nodes
    global node_cx, node_cy, node_cz, node_size, node_mass
    global node_com_x, node_com_y, node_com_z, node_child
    global node_particle, node_is_leaf
    global node_xmin, node_xmax, node_ymin, node_ymax, node_zmin, node_zmax
    
    df = pd.read_csv(filename)
    n = len(df)
    # init body arrays
    mass = [0.0] * n
    px = [0.0] * n
    py = [0.0] * n
    pz = [0.0] * n
    vx = [0.0] * n
    vy = [0.0] * n
    vz = [0.0] * n
    ax = [0.0] * n
    ay = [0.0] * n
    az = [0.0] * n
    
    # fill from dataframe
    for i in range(n):
        mass[i] = df.loc[i, 'mass']
        px[i] = df.loc[i, 'distanceX']
        py[i] = df.loc[i, 'distanceY']
        pz[i] = df.loc[i, 'distanceZ']
        vx[i] = df.loc[i, 'velocityX']
        vy[i] = df.loc[i, 'velocityY']
        vz[i] = df.loc[i, 'velocityZ']
    
    # set up node pool size (safe upper bound for octree)
    max_nodes = 8 * n + 100
    node_cx = [0.0] * max_nodes
    node_cy = [0.0] * max_nodes
    node_cz = [0.0] * max_nodes
    node_size = [0.0] * max_nodes
    node_mass = [0.0] * max_nodes
    node_com_x = [0.0] * max_nodes
    node_com_y = [0.0] * max_nodes
    node_com_z = [0.0] * max_nodes
    node_child = [-1] * (max_nodes * 8)
    node_particle = [-1] * max_nodes
    node_is_leaf = [1] * max_nodes
    node_xmin = [0.0] * max_nodes
    node_xmax = [0.0] * max_nodes
    node_ymin = [0.0] * max_nodes
    node_ymax = [0.0] * max_nodes
    node_zmin = [0.0] * max_nodes
    node_zmax = [0.0] * max_nodes

# correctness_check
# compare scalar vs numpy forces on same tree, report max error
# -----------------------------------------------------------------------
def correctness_check():
    print("--- correctness check: scalar vs numpy forces ---")
    root = build_tree()
 
    # scalar forces
    calculate_forces_scalar(root)
    scalar_ax = ax[:]
    scalar_ay = ay[:]
    scalar_az = az[:]
 
    # numpy bfs forces
    build_numpy_arrays()
    calculate_forces_bfs_numpy(root)
    numpy_ax = ax[:]
    numpy_ay = ay[:]
    numpy_az = az[:]
 
    # max absolute error across all particles and all axes
    max_err = 0.0
    for i in range(n):
        err = max(abs(scalar_ax[i] - numpy_ax[i]),
                  abs(scalar_ay[i] - numpy_ay[i]),
                  abs(scalar_az[i] - numpy_az[i]))
        if err > max_err:
            max_err = err
 
    # max magnitude for relative error
    max_mag = max(abs(scalar_ax[i]) for i in range(n))
 
    print(f"  max absolute error in acceleration: {max_err:.3e}")
    print(f"  max acceleration magnitude:         {max_mag:.3e}")
    if max_mag > 0:
        print(f"  max relative error:                 {max_err/max_mag:.3e}")
 
    if max_err < 1e-3 * max_mag:
        print("  PASS: numpy matches scalar")
    else:
        print("  FAIL: numpy does not match scalar - check implementation")
    print()
 

# benchmark
# time scalar vs numpy force calculation at multiple N values
def benchmark(filenames, num_trials=3):
    print("performance benchmark: scalar vs numpy BFS")
    print(f"{'N':>6}  {'scalar(s)':>10}  {'numpy(s)':>10}  {'speedup':>8}")
    print("-" * 42)
 
    for fname in filenames:
        load_csv(fname)
        root = build_tree()
        build_numpy_arrays()
 
        # time scalar
        t_scalar = 0.0
        for _ in range(num_trials):
            t0 = time.perf_counter()
            calculate_forces_scalar(root)
            t_scalar += time.perf_counter() - t0
        t_scalar /= num_trials
 
        # time numpy bfs
        t_numpy = 0.0
        for _ in range(num_trials):
            t0 = time.perf_counter()
            calculate_forces_bfs_numpy(root)
            t_numpy += time.perf_counter() - t0
        t_numpy /= num_trials
 
        speedup = t_scalar / t_numpy if t_numpy > 0 else 0
        print(f"{n:>6}  {t_scalar:>10.4f}  {t_numpy:>10.4f}  {speedup:>7.2f}x")
    print()
 

# run_simulation - main simulation loop
def run_tests():
    print("running correctness tests")

    # test 1: root mass = sum of all particle masses
    root = build_tree()
    total = sum(mass[:n])
    err = abs(node_mass[root] - total)
    assert err < 1e-3 * total, f"Fail: mass mismatch by {err:.3e}"
    print("Pass: mass is conserved")

    # test 2: every particle reachable from root
    found = [False] * n
    def traverse(node_idx):
        if node_is_leaf[node_idx]:
            b = node_particle[node_idx]
            if b != -1:
                found[b] = True
        else:
            for k in range(8):
                c = node_child[node_idx * 8 + k]
                if c != -1:
                    traverse(c)
    traverse(root)
    missing = [i for i,f in enumerate(found) if not f]
    assert len(missing) == 0, f"Fail: particles missing: {missing}"
    print("Pass: all particles found in tree")

    # test 3: root com matches direct calculation
    total_m = sum(mass[:n])
    cx_dir  = sum(mass[i]*px[i] for i in range(n)) / total_m
    cy_dir  = sum(mass[i]*py[i] for i in range(n)) / total_m
    cz_dir  = sum(mass[i]*pz[i] for i in range(n)) / total_m
    tol = 1e-6 * max(abs(cx_dir), abs(cy_dir), abs(cz_dir), 1.0)
    assert abs(node_com_x[root] - cx_dir) < tol, "Fail: com x mismatch"
    assert abs(node_com_y[root] - cy_dir) < tol, "Fail: com y mismatch"
    assert abs(node_com_z[root] - cz_dir) < tol, "Fail: com z mismatch"
    print("Pass: centre of mass is correct")

    # test 4: forces are finite (no NaN or Inf)
    calculate_forces_scalar(root)
    for i in range(n):
        assert math.isfinite(ax[i]), f"Fail: ax[{i}] is NaN/Inf"
        assert math.isfinite(ay[i]), f"Fail: ay[{i}] is NaN/Inf"
        assert math.isfinite(az[i]), f"Fail: az[{i}] is NaN/Inf"
    print("Pass: all forces finite")
    print("all tests passed\n")


# run_simulation: main simulation loop
def run_simulation(csv_filename, dt_val=dt, num_steps=steps):
    load_csv(csv_filename)
    print(f"loaded {n} bodies from {csv_filename}")

    run_tests()
    correctness_check()

    root = build_tree()
    build_numpy_arrays()
    calculate_forces_bfs_numpy(root)
    e0 = calculate_energy()
    print(f"initial energy: {e0:.6e}\n")

    for step in range(1, num_steps + 1):
        kick_half_step(dt_val)
        drift(dt_val)
        np_px[:] = px
        np_py[:] = py
        np_pz[:] = pz
        root = build_tree()
        build_numpy_arrays()
        calculate_forces_bfs_numpy(root)
        kick_half_step(dt_val)

        if step % 10 == 0:
            test(step)

    ef = calculate_energy()
    drift_pct = (ef - e0) / abs(e0) * 100.0
    print(f"final energy: {ef:.6e}")
    print(f"energy drift: {drift_pct:.6f}%")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python barnes_hut_numpy.py <csv_file> [steps]")
        print("python barnes_hut_numpy.py --benchmark <file1> <file2> ...")
        sys.exit(1)

    if sys.argv[1] == "--benchmark":
        benchmark(sys.argv[2:])
    else:
        filename  = sys.argv[1]
        num_steps = int(sys.argv[2]) if len(sys.argv) > 2 else steps
        run_simulation(filename, dt, num_steps)