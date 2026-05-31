# barnes-hut n-body simulation — morton-sorted particles (Warren & Salmon 1993)


import sys
import math
import numpy as np
import pandas as pd

# constants
G         = 6.67430e-11
softening = 1e9
theta     = 0.5
dt        = 8640.0    # seconds (0.1 day)
steps     = 100

# global node pool
max_nodes     = None
node_cx       = None
node_cy       = None
node_cz       = None
node_size     = None
node_mass     = None
node_com_x    = None
node_com_y    = None
node_com_z    = None
node_child    = None   # flat [max_nodes*8], -1 = no child
node_particle = None   # body index if leaf, else -1
node_is_leaf  = None   # 1 = leaf, 0 = internal
node_xmin = None; node_xmax = None
node_ymin = None; node_ymax = None
node_zmin = None; node_zmax = None

node_count = 0

# body arrays
n    = 0
mass = None
px = None; py = None; pz = None
vx = None; vy = None; vz = None
ax = None; ay = None; az = None


# Node pool helpers

def allocate_node():
    global node_count
    idx = node_count
    node_count += 1
    node_mass[idx]     = 0.0
    node_com_x[idx]    = 0.0
    node_com_y[idx]    = 0.0
    node_com_z[idx]    = 0.0
    node_particle[idx] = -1
    node_is_leaf[idx]  = 1
    return idx


def get_octant(node_idx, px_val, py_val, pz_val):
    oct_val = 0
    if px_val > node_cx[node_idx]: oct_val |= 4
    if py_val > node_cy[node_idx]: oct_val |= 2
    if pz_val > node_cz[node_idx]: oct_val |= 1
    return oct_val


def subdivide(node_idx):
    cx   = node_cx[node_idx]
    cy   = node_cy[node_idx]
    cz   = node_cz[node_idx]
    half = node_size[node_idx] / 2.0
    qtr  = half / 2.0

    child_centers = []
    for dx in (-qtr, qtr):
        for dy in (-qtr, qtr):
            for dz in (-qtr, qtr):
                child_centers.append((cx + dx, cy + dy, cz + dz))

    for octant, (ccx, ccy, ccz) in enumerate(child_centers):
        child_idx = allocate_node()
        node_cx[child_idx]   = ccx
        node_cy[child_idx]   = ccy
        node_cz[child_idx]   = ccz
        node_size[child_idx] = half
        ch = half / 2.0
        node_xmin[child_idx] = ccx - ch; node_xmax[child_idx] = ccx + ch
        node_ymin[child_idx] = ccy - ch; node_ymax[child_idx] = ccy + ch
        node_zmin[child_idx] = ccz - ch; node_zmax[child_idx] = ccz + ch
        node_child[node_idx * 8 + octant] = child_idx

    node_is_leaf[node_idx]  = 0
    node_particle[node_idx] = -1


def insert_particle(body_idx, node_idx):
    if node_is_leaf[node_idx]:
        if node_particle[node_idx] == -1:
            node_particle[node_idx] = body_idx
            node_mass[node_idx]     = mass[body_idx]
            node_com_x[node_idx]    = px[body_idx]
            node_com_y[node_idx]    = py[body_idx]
            node_com_z[node_idx]    = pz[body_idx]
        else:
            existing = node_particle[node_idx]
            subdivide(node_idx)
            insert_particle(existing, node_idx)
            insert_particle(body_idx, node_idx)
    else:
        oct       = get_octant(node_idx, px[body_idx], py[body_idx], pz[body_idx])
        child_idx = node_child[node_idx * 8 + oct]
        if child_idx == -1:
            raise RuntimeError("child missing in internal node")
        insert_particle(body_idx, child_idx)


def compute_mass(node_idx):
    if node_is_leaf[node_idx]:
        b = node_particle[node_idx]
        if b != -1:
            node_mass[node_idx]  = mass[b]
            node_com_x[node_idx] = px[b]
            node_com_y[node_idx] = py[b]
            node_com_z[node_idx] = pz[b]
        else:
            node_mass[node_idx] = 0.0
        return

    total_m = sum_cx = sum_cy = sum_cz = 0.0
    for k in range(8):
        child = node_child[node_idx * 8 + k]
        if child != -1:
            compute_mass(child)
            m = node_mass[child]
            if m > 0:
                total_m += m
                sum_cx  += node_com_x[child] * m
                sum_cy  += node_com_y[child] * m
                sum_cz  += node_com_z[child] * m

    node_mass[node_idx] = total_m
    if total_m > 0:
        node_com_x[node_idx] = sum_cx / total_m
        node_com_y[node_idx] = sum_cy / total_m
        node_com_z[node_idx] = sum_cz / total_m
    else:
        node_com_x[node_idx] = node_com_y[node_idx] = node_com_z[node_idx] = 0.0


-
def compute_morton_keys_vectorized(xs, ys, zs):
    """
    Compute Morton (Z-curve) keys for all particles at once.

    Uses cubic normalization: all three axes share the same scale factor
    (the longest axis span), so the key reflects true 3D proximity.

    Returns a numpy int64 array of length N.
    """
    xs = np.asarray(xs, dtype=np.float64)
    ys = np.asarray(ys, dtype=np.float64)
    zs = np.asarray(zs, dtype=np.float64)

    x_min = xs.min();  x_max = xs.max()
    y_min = ys.min();  y_max = ys.max()
    z_min = zs.min();  z_max = zs.max()

    max_range = max(x_max - x_min, y_max - y_min, z_max - z_min, 1e-9)

    BITS    = 21
    MAX_INT = (1 << BITS) - 1   # 2097151

    xi = np.clip(((xs - x_min) / max_range * MAX_INT).astype(np.int64), 0, MAX_INT)
    yi = np.clip(((ys - y_min) / max_range * MAX_INT).astype(np.int64), 0, MAX_INT)
    zi = np.clip(((zs - z_min) / max_range * MAX_INT).astype(np.int64), 0, MAX_INT)

    keys = np.zeros(len(xs), dtype=np.int64)
    for bit in range(BITS):
        keys |= ((xi >> bit) & 1) << (3 * bit)
        keys |= ((yi >> bit) & 1) << (3 * bit + 1)
        keys |= ((zi >> bit) & 1) << (3 * bit + 2)

    return keys


def sort_bodies_morton():
  
    global px, py, pz, vx, vy, vz, ax, ay, az, mass

    if n == 0:
        return

    keys  = compute_morton_keys_vectorized(px, py, pz)
    order = np.argsort(keys, kind='stable')  # stable preserves ties

    # numpy fancy-index then convert back to list
    px_a = np.array(px);   py_a = np.array(py);   pz_a = np.array(pz)
    vx_a = np.array(vx);   vy_a = np.array(vy);   vz_a = np.array(vz)
    ax_a = np.array(ax);   ay_a = np.array(ay);   az_a = np.array(az)
    m_a  = np.array(mass)

    px[:] = px_a[order].tolist();  py[:] = py_a[order].tolist();  pz[:] = pz_a[order].tolist()
    vx[:] = vx_a[order].tolist();  vy[:] = vy_a[order].tolist();  vz[:] = vz_a[order].tolist()
    ax[:] = ax_a[order].tolist();  ay[:] = ay_a[order].tolist();  az[:] = az_a[order].tolist()
    mass[:] = m_a[order].tolist()


# Interaction list (iterative — avoids Python recursion limit)
def get_interaction_list(body_idx, root, theta_val):
    ilist = []
    stack = [root]
    bx, by, bz = px[body_idx], py[body_idx], pz[body_idx]

    while stack:
        node_idx = stack.pop()

        if node_mass[node_idx] == 0.0:
            continue
        if node_is_leaf[node_idx] and node_particle[node_idx] == body_idx:
            continue

        dx = node_com_x[node_idx] - bx
        dy = node_com_y[node_idx] - by
        dz = node_com_z[node_idx] - bz
        r  = math.sqrt(dx*dx + dy*dy + dz*dz)

        if node_is_leaf[node_idx] or (r > 0 and node_size[node_idx] / r < theta_val):
            ilist.append(node_idx)
        else:
            for k in range(8):
                child = node_child[node_idx * 8 + k]
                if child != -1:
                    stack.append(child)

    return ilist


# Vectorised force kernel
def compute_force_numpy(i, ilist_np, px_np, py_np, pz_np,
                        soft, ncx, ncy, ncz, nm):
    dx = ncx[ilist_np] - px_np[i]
    dy = ncy[ilist_np] - py_np[i]
    dz = ncz[ilist_np] - pz_np[i]

    dist_sq   = dx*dx + dy*dy + dz*dz + soft*soft
    inv_dist3 = dist_sq ** -1.5

    f = G * nm[ilist_np] * inv_dist3

    return np.sum(f * dx), np.sum(f * dy), np.sum(f * dz)


def calculate_forces_bh_vectorized(root):
    global ax, ay, az

    px_np  = np.array(px)
    py_np  = np.array(py)
    pz_np  = np.array(pz)
    ncx_np = np.array(node_com_x)
    ncy_np = np.array(node_com_y)
    ncz_np = np.array(node_com_z)
    nm_np  = np.array(node_mass)

    for i in range(n):
        ilist = get_interaction_list(i, root, theta)
        if ilist:
            ilist_np = np.array(ilist, dtype=np.int32)
            ax[i], ay[i], az[i] = compute_force_numpy(
                i, ilist_np, px_np, py_np, pz_np,
                softening, ncx_np, ncy_np, ncz_np, nm_np)
        else:
            ax[i] = ay[i] = az[i] = 0.0


# Tree build
def build_tree():
    global node_count

    node_count = 0
    node_mass[:max_nodes]     = [0.0] * max_nodes
    node_com_x[:max_nodes]    = [0.0] * max_nodes
    node_com_y[:max_nodes]    = [0.0] * max_nodes
    node_com_z[:max_nodes]    = [0.0] * max_nodes
    node_particle[:max_nodes] = [-1]  * max_nodes
    node_is_leaf[:max_nodes]  = [1]   * max_nodes
    node_child[:max_nodes*8]  = [-1]  * (max_nodes * 8)

    cx, cy, cz, size = compute_bounding_box()
    root = allocate_node()
    node_cx[root]   = cx
    node_cy[root]   = cy
    node_cz[root]   = cz
    node_size[root] = size

    h = size / 2.0
    node_xmin[root] = cx - h; node_xmax[root] = cx + h
    node_ymin[root] = cy - h; node_ymax[root] = cy + h
    node_zmin[root] = cz - h; node_zmax[root] = cz + h

    for i in range(n):
        insert_particle(i, root)

    compute_mass(root)
    return root


def compute_bounding_box():
    min_x = min(px); max_x = max(px)
    min_y = min(py); max_y = max(py)
    min_z = min(pz); max_z = max(pz)

    cx = (min_x + max_x) / 2.0
    cy = (min_y + max_y) / 2.0
    cz = (min_z + max_z) / 2.0

    max_w   = max(max_x-min_x, max_y-min_y, max_z-min_z)
    padding = max_w * 0.01
    size    = (max_w if max_w > 0 else 1e10) + 2 * padding
    return cx, cy, cz, size


# Leapfrog KDK integrator
def kick_half_step(dt_val):
    for i in range(n):
        vx[i] += 0.5 * ax[i] * dt_val
        vy[i] += 0.5 * ay[i] * dt_val
        vz[i] += 0.5 * az[i] * dt_val

def drift(dt_val):
    for i in range(n):
        px[i] += vx[i] * dt_val
        py[i] += vy[i] * dt_val
        pz[i] += vz[i] * dt_val


# Diagnostics
def calculate_energy():
    kin = 0.0
    for i in range(n):
        kin += 0.5 * mass[i] * (vx[i]**2 + vy[i]**2 + vz[i]**2)
    pot = 0.0
    for i in range(n):
        for j in range(i+1, n):
            dx = px[i]-px[j]; dy = py[i]-py[j]; dz = pz[i]-pz[j]
            r  = math.sqrt(dx*dx + dy*dy + dz*dz)
            pot -= G * mass[i] * mass[j] / r
    return kin + pot

def print_state(step):
    total_m = sum(mass)
    cx = sum(mass[i]*px[i] for i in range(n)) / total_m
    cy = sum(mass[i]*py[i] for i in range(n)) / total_m
    cz = sum(mass[i]*pz[i] for i in range(n)) / total_m
    mx = sum(mass[i]*vx[i] for i in range(n))
    my = sum(mass[i]*vy[i] for i in range(n))
    mz = sum(mass[i]*vz[i] for i in range(n))
    e  = calculate_energy()
    print(f"step {step:4d}: E={e:.4e}  CoM=({cx:.3e},{cy:.3e},{cz:.3e})"
          f"  mom=({mx:.3e},{my:.3e},{mz:.3e})")
    for i in range(min(3, n)):
        print(f"  body {i}: pos=({px[i]:.3e},{py[i]:.3e},{pz[i]:.3e})"
              f"  vel=({vx[i]:.3e},{vy[i]:.3e},{vz[i]:.3e})")


# CSV loader
def load_csv(filename):
    global n, mass, px, py, pz, vx, vy, vz, ax, ay, az, max_nodes
    global node_cx, node_cy, node_cz, node_size, node_mass
    global node_com_x, node_com_y, node_com_z, node_child
    global node_particle, node_is_leaf
    global node_xmin, node_xmax, node_ymin, node_ymax, node_zmin, node_zmax

    df = pd.read_csv(filename)
    n  = len(df)
    print("Columns:", list(df.columns))

    mass = [float(v) for v in df['mass']]
    px   = [float(v) for v in df['distanceX']]
    py   = [float(v) for v in df['distanceY']]
    pz   = [float(v) for v in df['distanceZ']]
    vx   = [float(v) for v in df['velocityX']]
    vy   = [float(v) for v in df['velocityY']]
    vz   = [float(v) for v in df['velocityZ']]
    ax   = [0.0] * n;  ay = [0.0] * n;  az = [0.0] * n

    max_nodes    = 8 * n + 100
    node_cx      = [0.0] * max_nodes
    node_cy      = [0.0] * max_nodes
    node_cz      = [0.0] * max_nodes
    node_size    = [0.0] * max_nodes
    node_mass    = [0.0] * max_nodes
    node_com_x   = [0.0] * max_nodes
    node_com_y   = [0.0] * max_nodes
    node_com_z   = [0.0] * max_nodes
    node_child   = [-1]  * (max_nodes * 8)
    node_particle= [-1]  * max_nodes
    node_is_leaf = [1]   * max_nodes
    node_xmin    = [0.0] * max_nodes;  node_xmax = [0.0] * max_nodes
    node_ymin    = [0.0] * max_nodes;  node_ymax = [0.0] * max_nodes
    node_zmin    = [0.0] * max_nodes;  node_zmax = [0.0] * max_nodes


# Main simulation loop
def run_simulation(csv_filename, dt_val=dt, n_steps=steps):
    load_csv(csv_filename)
    print(f"Loaded {n} bodies from {csv_filename}")

    sort_bodies_morton()
    root = build_tree()
    calculate_forces_bh_vectorized(root)
    E0 = calculate_energy()
    print(f"Initial total energy: {E0:.6e}")

    for step in range(1, n_steps + 1):
        kick_half_step(dt_val)
        drift(dt_val)

        sort_bodies_morton()   # reorder after positions change
        root = build_tree()
        calculate_forces_bh_vectorized(root)
        kick_half_step(dt_val)

        if step % 10 == 0:
            print_state(step)

    Ef = calculate_energy()
    print(f"Final total energy:  {Ef:.6e}")
    print(f"Energy drift:        {(Ef - E0)/abs(E0)*100:.6f} %")


# Tests
def test_mass_conservation():
    root      = build_tree()
    tree_mass = node_mass[root]
    exact     = sum(mass)
    rel_err   = abs(tree_mass - exact) / exact
    print(f"root mass: {tree_mass:.6e},  particle sum: {exact:.6e},  rel_err={rel_err:.2e}")
    assert rel_err < 1e-5, f"FAIL: mass conservation rel_err={rel_err}"
    print("test_mass_conservation: PASS")


def test_all_particles_found():
    root  = build_tree()
    found = [False] * n

    def traverse(node_idx):
        if node_is_leaf[node_idx]:
            if node_particle[node_idx] != -1:
                found[node_particle[node_idx]] = True
        else:
            for k in range(8):
                child = node_child[node_idx * 8 + k]
                if child != -1:
                    traverse(child)

    traverse(root)
    missing = [i for i, f in enumerate(found) if not f]
    assert not missing, f"FAIL: particles not found: {missing}"
    print("test_all_particles_found: PASS")


def test_center_of_mass():
    root    = build_tree()
    total_m = sum(mass)
    dir_cx  = sum(mass[i]*px[i] for i in range(n)) / total_m
    dir_cy  = sum(mass[i]*py[i] for i in range(n)) / total_m
    dir_cz  = sum(mass[i]*pz[i] for i in range(n)) / total_m

    print(f"tree   CoM: ({node_com_x[root]:.6e}, {node_com_y[root]:.6e}, {node_com_z[root]:.6e})")
    print(f"direct CoM: ({dir_cx:.6e}, {dir_cy:.6e}, {dir_cz:.6e})")

    REL_TOL = 1e-9
    for label, tv, dv in (('x', node_com_x[root], dir_cx),
                           ('y', node_com_y[root], dir_cy),
                           ('z', node_com_z[root], dir_cz)):
        scale   = abs(dv) if abs(dv) > 0 else 1.0
        rel_err = abs(tv - dv) / scale
        assert rel_err < REL_TOL, f"FAIL: CoM mismatch {label}: rel_err={rel_err:.2e}"
    print("test_center_of_mass: PASS")


def test_bounding_boxes():
    root = build_tree()

    def check(node_idx):
        if node_is_leaf[node_idx]:
            b = node_particle[node_idx]
            if b != -1:
                assert node_xmin[node_idx] <= px[b] <= node_xmax[node_idx], \
                    f"FAIL: particle {b} out of x bounds"
                assert node_ymin[node_idx] <= py[b] <= node_ymax[node_idx], \
                    f"FAIL: particle {b} out of y bounds"
                assert node_zmin[node_idx] <= pz[b] <= node_zmax[node_idx], \
                    f"FAIL: particle {b} out of z bounds"
        else:
            for k in range(8):
                child = node_child[node_idx * 8 + k]
                if child != -1:
                    check(child)

    check(root)
    print("test_bounding_boxes: PASS")


def test_regression_one_step():
    sort_bodies_morton()
    root = build_tree()
    calculate_forces_bh_vectorized(root)
    kick_half_step(dt)
    drift(dt)

    sort_bodies_morton()
    root = build_tree()
    calculate_forces_bh_vectorized(root)
    kick_half_step(dt)

    print("Positions after one step:")
    for i in range(min(5, n)):
        print(f"  body {i}: ({px[i]:.3e}, {py[i]:.3e}, {pz[i]:.3e})")

    nan_bodies = [i for i in range(n) if math.isnan(px[i])]
    assert not nan_bodies, f"FAIL: NaN in positions for bodies {nan_bodies}"
    print("test_regression_one_step: PASS")


def test_morton_locality():
    """
    Verify that Morton sort improves spatial locality.
    After sorting, adjacent particles in the array should be closer
    together in 3D space on average than before sorting.
    """
    px_orig = px[:]; py_orig = py[:]; pz_orig = pz[:]

    def mean_adjacent_dist(pxl, pyl, pzl):
        dists = []
        for i in range(len(pxl)-1):
            dx = pxl[i+1]-pxl[i]; dy = pyl[i+1]-pyl[i]; dz = pzl[i+1]-pzl[i]
            dists.append(math.sqrt(dx*dx + dy*dy + dz*dz))
        return sum(dists)/len(dists)

    dist_before = mean_adjacent_dist(px_orig, py_orig, pz_orig)
    sort_bodies_morton()
    dist_after  = mean_adjacent_dist(px, py, pz)

    print(f"Mean adjacent distance:  before={dist_before:.3e}  after={dist_after:.3e}"
          f"  improvement={dist_before/dist_after:.2f}x")
    assert dist_after < dist_before, \
        f"FAIL: Morton sort did not improve spatial locality"
    print("test_morton_locality: PASS")

    # restore original order so other tests are unaffected
    px[:] = px_orig; py[:] = py_orig; pz[:] = pz_orig


# Entry point
if __name__ == "__main__":
    if len(sys.argv) > 1:
        run_simulation(sys.argv[1])
    else:
        # MORTON-FIX-3: was r"CSV files\stable_random_system100.csv"
        test_file = "CSV files\stable_random_system100.csv"

        print(f"Loading: {test_file}")
        load_csv(test_file)
        print("Running tests...\n")

        test_mass_conservation()
        test_all_particles_found()
        test_center_of_mass()
        test_bounding_boxes()
        test_morton_locality()
        test_regression_one_step()

        print("\nAll tests passed.")