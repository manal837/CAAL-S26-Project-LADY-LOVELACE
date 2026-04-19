# barnes-hut n-body simulation - refactored python version

import sys
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# constants
G = 6.67430e-11
softening = 1e9
theta = 0.5
dt = 8640.0          # time step in seconds, match c++ example
steps = 100

# global node pool (sized after reading n)
# we set max_nodes = 8 * n + 100
max_nodes = none

# node pool arrays - flat arrays for all node fields
node_cx = none      # center x of node
node_cy = none
node_cz = none
node_size = none    # side length of cube
node_mass = none
node_com_x = none
node_com_y = none
node_com_z = none
node_child = none   # flat array size max_nodes * 8, -1 means no child
node_particle = none # particle index if leaf, else -1
node_is_leaf = none  # 1 if leaf, 0 if internal
node_xmin = none
node_xmax = none
node_ymin = none
node_ymax = none
node_zmin = none
node_zmax = none

node_count = 0      # next free node index

# body arrays (global for simplicity, matches assembly layout)
n = 0
mass = none
px = none
py = none
pz = none
vx = none
vy = none
vz = none
ax = none
ay = none
az = none


# allocate_node - get next free node from pool
# returns: idx = new node index
# increments node_count and returns previous value

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


# get_octant - figure out which of 8 octants a point falls into
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


# subdivide - split a leaf node into 8 children (one per octant)
# args: node_idx - the node to subdivide (must be leaf)
# does: allocates 8 children, sets their centers, sizes, bounding boxes
#       marks parent as internal (is_leaf = 0)

def subdivide(node_idx):
    global node_count
    
    # get parent bounding box
    xmin = node_xmin[node_idx]
    xmax = node_xmax[node_idx]
    ymin = node_ymin[node_idx]
    ymax = node_ymax[node_idx]
    zmin = node_zmin[node_idx]
    zmax = node_zmax[node_idx]
    
    # get parent center
    cx = node_cx[node_idx]
    cy = node_cy[node_idx]
    cz = node_cz[node_idx]
    
    # child size is half of parent size
    half = node_size[node_idx] / 2.0
    quarter = half / 2.0      # child half-extent for bounding box
    
    # loop over all 8 combos of dx, dy, dz offsets
    # order: (-,-,-), (-,-,+), (-,+,-), (-,+,+), (+,-,-), (+,-,+), (+,+,-), (+,+,+)
    child_centers = []
    for dz in (-quarter, quarter):
        for dy in (-quarter, quarter):
            for dx in (-quarter, quarter):
                child_centers.append((cx + dx, cy + dy, cz + dz))
    
    # create each child node
    for octant, (ccx, ccy, ccz) in enumerate(child_centers):
        child_idx = allocate_node()
        
        # store child center
        node_cx[child_idx] = ccx
        node_cy[child_idx] = ccy
        node_cz[child_idx] = ccz
        node_size[child_idx] = half
        
        # child bounding box = [center - quarter, center + quarter]
        child_half = half / 2.0
        node_xmin[child_idx] = ccx - child_half
        node_xmax[child_idx] = ccx + child_half
        node_ymin[child_idx] = ccy - child_half
        node_ymax[child_idx] = ccy + child_half
        node_zmin[child_idx] = ccz - child_half
        node_zmax[child_idx] = ccz + child_half
        
        # store child pointer in parent (flat array access)
        node_child[node_idx * 8 + octant] = child_idx
    
    # mark parent as internal (no longer a leaf)
    node_is_leaf[node_idx] = 0
    node_particle[node_idx] = -1


# insert_particle - insert a body into the octree
# uses recursion: goes down tree until finds empty leaf
# if leaf occupied, subdivide then re-insert both bodies

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
            # occupied leaf - subdivide and re-insert both
            subdivide(node_idx)
            # re-insert existing body that was here
            existing = node_particle[node_idx]  # old body index
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
            raise runtimeerror("child missing in internal node")
        insert_particle(body_idx, child_idx)


# compute_mass - post-order traversal to compute mass and com
# args: node_idx - current node
# does: for leaves, mass/com already set from insert
#       for internal nodes, sum over children then divide

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
# does: if node is far (size/dist < theta) treat as point mass
#       else recurse into children

def calculate_force_bh(body_idx, node_idx, theta):
    # skip zero mass nodes
    if node_mass[node_idx] == 0.0:
        return
    
    # skip self-interaction (leaf containing same body)
    if node_is_leaf[node_idx] and node_particle[node_idx] == body_idx:
        return
    
    # compute vector from body to node's center of mass
    dx = node_com_x[node_idx] - px[body_idx]
    dy = node_com_y[node_idx] - py[body_idx]
    dz = node_com_z[node_idx] - pz[body_idx]
    r = math.sqrt(dx*dx + dy*dy + dz*dz)
    
    # barnes-hut criterion: check if node is far enough
    if node_is_leaf[node_idx] or (node_size[node_idx] / r < theta):
        # treat node as point mass
        r_soft = math.sqrt(r*r + softening*softening)
        r_cubed = r_soft * r_soft * r_soft
        factor = g * node_mass[node_idx] / r_cubed
        
        # add acceleration to body
        ax[body_idx] += factor * dx
        ay[body_idx] += factor * dy
        az[body_idx] += factor * dz
    else:
        # not far enough - recurse into all children
        for k in range(8):
            child = node_child[node_idx * 8 + k]
            if child != -1:
                calculate_force_bh(body_idx, child, theta)


# compute_bounding_box - find global bounding box of all bodies
# returns: cx, cy, cz (center), size (cube side with 1% padding)

def compute_bounding_box():
    global n, px, py, pz
    
    min_x = min(px)
    max_x = max(px)
    min_y = min(py)
    max_y = max(py)
    min_z = min(pz)
    max_z = max(pz)
    
    # center of bounding box
    cx = (min_x + max_x) / 2.0
    cy = (min_y + max_y) / 2.0
    cz = (min_z + max_z) / 2.0
    
    # size = max width + 2% padding
    width_x = max_x - min_x
    width_y = max_y - min_y
    width_z = max_z - min_z
    max_width = max(width_x, width_y, width_z)
    padding = max_width * 0.01
    size = max_width + 2 * padding
    
    if size <= 0:
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
    
    # compute bounding box and create root node
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


# calculate_forces_bh - wrapper: zero accel then compute all forces
def calculate_forces_bh(root):
    global ax, ay, az
    
    # zero all accelerations
    for i in range(n):
        ax[i] = 0.0
        ay[i] = 0.0
        az[i] = 0.0
    
    # compute force on each body
    for i in range(n):
        calculate_force_bh(i, root, theta)


# kick_half_step - leapfrog half-kick: v += a * dt/2
def kick_half_step(dt):
    for i in range(n):
        vx[i] += 0.5 * ax[i] * dt
        vy[i] += 0.5 * ay[i] * dt
        vz[i] += 0.5 * az[i] * dt


# drift - leapfrog drift: p += v * dt
def drift(dt):
    for i in range(n):
        px[i] += vx[i] * dt
        py[i] += vy[i] * dt
        pz[i] += vz[i] * dt


# calculate_energy - compute total energy (kinetic + potential)
# returns: total energy in joules
def calculate_energy():
    kin = 0.0
    pot = 0.0
    
    # kinetic energy = 1/2 * m * v^2
    for i in range(n):
        v2 = vx[i]*vx[i] + vy[i]*vy[i] + vz[i]*vz[i]
        kin += 0.5 * mass[i] * v2
    
    # potential energy = -g * m1 * m2 / r
    for i in range(n):
        for j in range(i+1, n):
            dx = px[i] - px[j]
            dy = py[i] - py[j]
            dz = pz[i] - pz[j]
            r = math.sqrt(dx*dx + dy*dy + dz*dz)
            pot -= g * mass[i] * mass[j] / r
    
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
    
    print(f"step {step}: energy = {energy:.6e}, com = ({com_x:.3e}, {com_y:.3e}, {com_z:.3e}), momentum = ({mom_x:.3e}, {mom_y:.3e}, {mom_z:.3e})")
    
    for i in range(min(3, n)):
        print(f"  body {i}: pos=({px[i]:.3e}, {py[i]:.3e}, {pz[i]:.3e}) vel=({vx[i]:.3e}, {vy[i]:.3e}, {vz[i]:.3e})")


# load_csv - read body data from csv file into global arrays
def load_csv(filename):
    global n, mass, px, py, pz, vx, vy, vz, ax, ay, az, max_nodes
    global node_cx, node_cy, node_cz, node_size, node_mass
    global node_com_x, node_com_y, node_com_z, node_child
    global node_particle, node_is_leaf, node_xmin, node_xmax
    global node_ymin, node_ymax, node_zmin, node_zmax
    
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
        px[i] = df.loc[i, 'distancex']
        py[i] = df.loc[i, 'distancey']
        pz[i] = df.loc[i, 'distancez']
        vx[i] = df.loc[i, 'velocityx']
        vy[i] = df.loc[i, 'velocityy']
        vz[i] = df.loc[i, 'velocityz']
    
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


# run_simulation - main simulation loop

def run_simulation(csv_filename, dt=dt, steps=steps):
    load_csv(csv_filename)
    print(f"loaded {n} bodies from {csv_filename}")
    
    # build tree and compute initial forces
    root = build_tree()
    calculate_forces_bh(root)
    initial_energy = calculate_energy()
    print(f"initial total energy: {initial_energy:.6e}")
    
    # time stepping loop
    for step in range(1, steps + 1):
        kick_half_step(dt)
        drift(dt)
        root = build_tree()          # rebuild tree each step
        calculate_forces_bh(root)
        kick_half_step(dt)
        
        if step % 10 == 0:
            test(step)
    
    final_energy = calculate_energy()
    print(f"final total energy: {final_energy:.6e}")
    print(f"energy drift: {(final_energy - initial_energy)/abs(initial_energy)*100:.6f}%")


def test_mass_conservation():
    """check that root mass equals sum of particle masses."""
    root = build_tree()
    total_particle_mass = sum(mass)
    print(f"root mass: {node_mass[root]}, total particle mass: {total_particle_mass}")
    assert abs(node_mass[root] - total_particle_mass) < 1e-9, "mass conservation failed"

def test_all_particles_found():
    """check that every particle is reachable from root."""
    root = build_tree()
    found = [false] * n
    
    def traverse(node_idx):
        if node_is_leaf[node_idx]:
            if node_particle[node_idx] != -1:
                found[node_particle[node_idx]] = true
        else:
            for k in range(8):
                child = node_child[node_idx * 8 + k]
                if child != -1:
                    traverse(child)
    
    traverse(root)
    assert all(found), f"particles not found: {[i for i, f in enumerate(found) if not f]}"
    print("all particles found in tree.")

def test_center_of_mass():
    """root com should match direct com calculation."""
    root = build_tree()
    total_mass = sum(mass)
    com_x_dir = sum(mass[i]*px[i] for i in range(n)) / total_mass
    com_y_dir = sum(mass[i]*py[i] for i in range(n)) / total_mass
    com_z_dir = sum(mass[i]*pz[i] for i in range(n)) / total_mass
    
    print(f"root com: ({node_com_x[root]}, {node_com_y[root]}, {node_com_z[root]})")
    print(f"direct com: ({com_x_dir}, {com_y_dir}, {com_z_dir})")
    
    assert abs(node_com_x[root] - com_x_dir) < 1e-9, "com mismatch x"
    assert abs(node_com_y[root] - com_y_dir) < 1e-9, "com mismatch y"
    assert abs(node_com_z[root] - com_z_dir) < 1e-9, "com mismatch z"

def test_bounding_boxes():
    """every particle lies within its leaf node's bounding box."""
    root = build_tree()
    
    def check(node_idx):
        if node_is_leaf[node_idx]:
            if node_particle[node_idx] != -1:
                b = node_particle[node_idx]
                x, y, z = px[b], py[b], pz[b]
                assert node_xmin[node_idx] <= x <= node_xmax[node_idx], f"particle {b} out of x bounds"
                assert node_ymin[node_idx] <= y <= node_ymax[node_idx], f"particle {b} out of y bounds"
                assert node_zmin[node_idx] <= z <= node_zmax[node_idx], f"particle {b} out of z bounds"
        else:
            for k in range(8):
                child = node_child[node_idx * 8 + k]
                if child != -1:
                    check(child)
    
    check(root)
    print("all particles inside leaf bounding boxes.")

def test_regression_one_step():
    """compare positions after one step (no nans, prints for manual verification)."""
    root = build_tree()
    calculate_forces_bh(root)
    kick_half_step(dt)
    drift(dt)
    calculate_forces_bh(root)
    kick_half_step(dt)
    
    print("positions after one step:")
    for i in range(min(5, n)):
        print(f"  body {i}: ({px[i]}, {py[i]}, {pz[i]})")
    
    # check for nans
    assert all(not math.isnan(px[i]) for i in range(n)), "nan in positions"

S
# main - run simulation or tests based on command line args
if __name__ == "__main__":
    if len(sys.argv) > 1:
        filename = sys.argv[1]
        run_simulation(filename)
    else:
        # run tests on default file
        test_file = "stable_random_system100.csv"
        print("loading test file:", test_file)
        load_csv(test_file)
        print("running tests...")
        test_mass_conservation()
        test_all_particles_found()
        test_center_of_mass()
        test_bounding_boxes()
        test_regression_one_step()
        print("all tests passed.")