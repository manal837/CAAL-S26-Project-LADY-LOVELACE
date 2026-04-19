.section .data
.align 2

.equ N,         100
.equ MAX_NODES, 900          # 8 * N + 100 = 800 + 100 = 900

# Make body arrays visible to C code
.globl px, py, pz
.globl vx, vy, vz
.globl ax, ay, az
.globl mass

# Body state arrays (using .rept .float .endr as in naive_nbody.S)
px:   .rept N .float 0.0 .endr
py:   .rept N .float 0.0 .endr
pz:   .rept N .float 0.0 .endr
vx:   .rept N .float 0.0 .endr
vy:   .rept N .float 0.0 .endr
vz:   .rept N .float 0.0 .endr
ax:   .rept N .float 0.0 .endr
ay:   .rept N .float 0.0 .endr
az:   .rept N .float 0.0 .endr
mass: .rept N .float 0.0 .endr

# Octree node pool - following PDF Listing 3 exactly with .rept/.endr
# Centre of mass (float arrays)
node_cx:   .rept MAX_NODES .float 0.0 .endr
node_cy:   .rept MAX_NODES .float 0.0 .endr
node_cz:   .rept MAX_NODES .float 0.0 .endr
node_mass: .rept MAX_NODES .float 0.0 .endr

# Children: 8 per node, stored flat. -1 means "no child"
# This is the KEY data structure from the PDF
node_child: .rept (MAX_NODES * 8) .word -1 .endr

# Particle index if leaf, else -1
node_particle: .rept MAX_NODES .word -1 .endr

# 1 = leaf, 0 = internal
node_is_leaf: .rept MAX_NODES .word 1 .endr

# Bounding box per node (required by PDF)
node_xmin: .rept MAX_NODES .float 0.0 .endr
node_xmax: .rept MAX_NODES .float 0.0 .endr
node_ymin: .rept MAX_NODES .float 0.0 .endr
node_ymax: .rept MAX_NODES .float 0.0 .endr
node_zmin: .rept MAX_NODES .float 0.0 .endr
node_zmax: .rept MAX_NODES .float 0.0 .endr

# Node size (side length of cube) - used for Barnes-Hut criterion
node_size: .rept MAX_NODES .float 0.0 .endr

# Centre of mass (separate from bounding box, as per PDF)
node_com_x: .rept MAX_NODES .float 0.0 .endr
node_com_y: .rept MAX_NODES .float 0.0 .endr
node_com_z: .rept MAX_NODES .float 0.0 .endr

node_count: .word 0

# Constants
softening_float: .float 1e11
G_float:         .float 6.67430e-11
dt_float:        .float 86400.0
theta_float:     .float 0.1
half_dt_float:   .float 43200.0

# Temporary storage
temp_min_x: .float 0.0
temp_max_x: .float 0.0
temp_min_y: .float 0.0
temp_max_y: .float 0.0
temp_min_z: .float 0.0
temp_max_z: .float 0.0
.section .text


# allocate_node
# gives us the next free node from the pool and increments the counter

#
# registers used:
#   t0 = address of node_count
#   a0 = current count (this becomes the new node index)
#   t1 = count + 1 (we write this back)


.globl allocate_node
allocate_node:
    la   t0, node_count       # t0 points to node_count variable
    lw   a0, 0(t0)            # a0 = current value of node_count
    addi t1, a0, 1            # t1 = node_count + 1
    sw   t1, 0(t0)            # store the incremented count back
    ret                       # return a0 as the new node index


# get_octant
# figures out which of the 8 octants a point falls into
# relative to the centre of a given node
#
# we encode the octant as a 3-bit number:
#   bit 2 (value 4): x >= cx
#   bit 1 (value 2): y >= cy
#   bit 0 (value 1): z >= cz


# registers used:
#   t0 = byte offset = node_index * 4
#   t1 = scratch address for loading
#   t3 = result of float comparison (0 or 1)
#   ft0 = node centre x
#   ft1 = node centre y
#   ft2 = node centre z


.globl get_octant
get_octant:
    slli t0, a0, 2            # t0 = node_index * 4 (byte offset for float arrays)

    la   t1, node_cx
    add  t1, t1, t0
    flw  ft0, 0(t1)           # ft0 = cx of this node

    la   t1, node_cy
    add  t1, t1, t0
    flw  ft1, 0(t1)           # ft1 = cy of this node

    la   t1, node_cz
    add  t1, t1, t0
    flw  ft2, 0(t1)           # ft2 = cz of this node

    li   a0, 0                # start with octant = 0 (all bits clear)

    flt.s t3, fa0, ft0        # t3 = 1 if point x < cx (left half)
    bnez  t3, 1f              # if x < cx leave bit2 = 0
    li    a0, 4               # else x >= cx so set bit2 (octant |= 4)
1:
    flt.s t3, fa1, ft1        # t3 = 1 if point y < cy (lower half)
    bnez  t3, 2f              # if y < cy leave bit1 = 0
    addi  a0, a0, 2           # else y >= cy so set bit1 (octant |= 2)
2:
    flt.s t3, fa2, ft2        # t3 = 1 if point z < cz (front half)
    bnez  t3, 3f              # if z < cz leave bit0 = 0
    addi  a0, a0, 1           # else z >= cz so set bit0 (octant |= 1)
3:
    ret                       # return octant 0-7 in a0


# subdivide
# splits a leaf node into 8 children (one per octant)
# each child gets its own centre, size, and bounding box


# saved registers (callee-saved, pushed to stack):
#   s0 = node index
#   s1 = child loop counter (0 to 7)
#   s2 = pointer into node_child array
#
# float registers:
#   ft0  = xmin then reused as child cx
#   ft1  = xmax then reused as child cy
#   ft2  = width_x then reused as child cz
#   ft3  = ymin then reused as bounding box scratch
#   ft4  = ymax then reused
#   ft5  = width_y
#   ft6  = zmin
#   ft7  = zmax
#   ft8  = width_z, then reused as quarter (child half-extent)
#   ft9  = max_size
#   ft10 = 2.0 constant
#   ft11 = half_size = max_size / 2
#   fa3  = parent cx (saved here so we can reuse ft0-ft2 in loop)
#   fa4  = parent cy
#   fa5  = parent cz


.globl subdivide
subdivide:
    addi sp, sp, -32
    sd   ra, 24(sp)
    sd   s0, 16(sp)
    sd   s1,  8(sp)
    sd   s2,  0(sp)

    mv   s0, a0               # s0 = node index, keep it safe

    slli t0, s0, 2            # t0 = node_index * 4 (byte offset)

    # load the bounding box of this node
    la   t1, node_xmin
    add  t1, t1, t0
    flw  ft0, 0(t1)           # ft0 = xmin

    la   t1, node_xmax
    add  t1, t1, t0
    flw  ft1, 0(t1)           # ft1 = xmax

    fsub.s ft2, ft1, ft0      # ft2 = width_x = xmax - xmin

    la   t1, node_ymin
    add  t1, t1, t0
    flw  ft3, 0(t1)           # ft3 = ymin

    la   t1, node_ymax
    add  t1, t1, t0
    flw  ft4, 0(t1)           # ft4 = ymax

    fsub.s ft5, ft4, ft3      # ft5 = width_y = ymax - ymin

    la   t1, node_zmin
    add  t1, t1, t0
    flw  ft6, 0(t1)           # ft6 = zmin

    la   t1, node_zmax
    add  t1, t1, t0
    flw  ft7, 0(t1)           # ft7 = zmax

    fsub.s ft8, ft7, ft6      # ft8 = width_z = zmax - zmin

    # make it a cube by using the max of all three widths
    fmax.s ft9, ft2, ft5      # ft9 = max(width_x, width_y)
    fmax.s ft9, ft9, ft8      # ft9 = max of all three = max_size

    # store this as the node size
    la   t1, node_size
    add  t1, t1, t0
    fsw  ft9, 0(t1)           # node_size[s0] = max_size

    # compute half_size and quarter
    li   t1, 2
    fcvt.s.w ft10, t1         # ft10 = 2.0 (convert int 2 to float)
    fdiv.s ft11, ft9, ft10    # ft11 = half_size = max_size / 2
    fdiv.s ft8, ft11, ft10    # ft8  = quarter   = half_size / 2

    # save parent centre in fa3/fa4/fa5 so we can reuse ft0-ft2 in loop
    la   t1, node_cx
    add  t1, t1, t0
    flw  fa3, 0(t1)           # fa3 = parent cx

    la   t1, node_cy
    add  t1, t1, t0
    flw  fa4, 0(t1)           # fa4 = parent cy

    la   t1, node_cz
    add  t1, t1, t0
    flw  fa5, 0(t1)           # fa5 = parent cz

    # point s2 at the start of node_child[s0 * 8]
    li   s1, 0                # s1 = child counter starting at 0
    la   s2, node_child       # s2 = base of child array
    slli t1, s0, 3            # t1 = s0 * 8
    slli t1, t1, 2            # t1 = s0 * 32 (byte offset: 8 children * 4 bytes each)
    add  s2, s2, t1           # s2 now points to child[s0*8]

child_loop:
    # compute child centre x: if bit2 of s1 is set -> +x half, else -x half
    andi t2, s1, 4            # t2 = s1 & 4 (check x bit)
    beqz t2, cx_neg
    fadd.s ft0, fa3, ft8      # ft0 = parent_cx + quarter (right half)
    j    cx_done
cx_neg:
    fsub.s ft0, fa3, ft8      # ft0 = parent_cx - quarter (left half)
cx_done:

    # compute child centre y: bit1 set -> +y half
    andi t2, s1, 2
    beqz t2, cy_neg
    fadd.s ft1, fa4, ft8      # ft1 = parent_cy + quarter
    j    cy_done
cy_neg:
    fsub.s ft1, fa4, ft8      # ft1 = parent_cy - quarter
cy_done:

    # compute child centre z: bit0 set -> +z half
    andi t2, s1, 1
    beqz t2, cz_neg
    fadd.s ft2, fa5, ft8      # ft2 = parent_cz + quarter
    j    cz_done
cz_neg:
    fsub.s ft2, fa5, ft8      # ft2 = parent_cz - quarter
cz_done:

    # allocate this child node
    call allocate_node        # a0 = new child node index
    mv   t3, a0               # t3 = child node index

    # store child index in parent's child array
    sw   t3, 0(s2)            # node_child[s0*8 + s1] = child index
    addi s2, s2, 4            # advance pointer to next child slot

    slli t4, t3, 2            # t4 = child_index * 4 (byte offset for child fields)

    # store child centre
    la   t5, node_cx
    add  t5, t5, t4
    fsw  ft0, 0(t5)           # node_cx[child] = child cx

    la   t5, node_cy
    add  t5, t5, t4
    fsw  ft1, 0(t5)           # node_cy[child] = child cy

    la   t5, node_cz
    add  t5, t5, t4
    fsw  ft2, 0(t5)           # node_cz[child] = child cz

    # child size is half of parent size
    la   t5, node_size
    add  t5, t5, t4
    fsw  ft11, 0(t5)          # node_size[child] = half_size

    # child bounding box x = [cx - quarter, cx + quarter]
    fsub.s ft3, ft0, ft8      # ft3 = cx - quarter = xmin
    fadd.s ft4, ft0, ft8      # ft4 = cx + quarter = xmax
    la   t5, node_xmin
    add  t5, t5, t4
    fsw  ft3, 0(t5)           # store xmin
    la   t5, node_xmax
    add  t5, t5, t4
    fsw  ft4, 0(t5)           # store xmax

    # child bounding box y
    fsub.s ft3, ft1, ft8      # ft3 = cy - quarter = ymin
    fadd.s ft4, ft1, ft8      # ft4 = cy + quarter = ymax
    la   t5, node_ymin
    add  t5, t5, t4
    fsw  ft3, 0(t5)
    la   t5, node_ymax
    add  t5, t5, t4
    fsw  ft4, 0(t5)

    # child bounding box z
    fsub.s ft3, ft2, ft8      # ft3 = cz - quarter = zmin
    fadd.s ft4, ft2, ft8      # ft4 = cz + quarter = zmax
    la   t5, node_zmin
    add  t5, t5, t4
    fsw  ft3, 0(t5)
    la   t5, node_zmax
    add  t5, t5, t4
    fsw  ft4, 0(t5)

    # mark child as empty leaf
    la   t5, node_is_leaf
    add  t5, t5, t4
    li   t6, 1
    sw   t6, 0(t5)            # node_is_leaf[child] = 1

    la   t5, node_particle
    add  t5, t5, t4
    li   t6, -1
    sw   t6, 0(t5)            # node_particle[child] = -1 (no body yet)

    # zero out child mass and centre of mass
    fcvt.s.w ft3, x0          # ft3 = 0.0
    la   t5, node_mass
    add  t5, t5, t4
    fsw  ft3, 0(t5)           # node_mass[child] = 0.0
    la   t5, node_com_x
    add  t5, t5, t4
    fsw  ft3, 0(t5)           # node_com_x[child] = 0.0
    la   t5, node_com_y
    add  t5, t5, t4
    fsw  ft3, 0(t5)           # node_com_y[child] = 0.0
    la   t5, node_com_z
    add  t5, t5, t4
    fsw  ft3, 0(t5)           # node_com_z[child] = 0.0

    addi s1, s1, 1            # s1++ move to next child
    li   t5, 8
    blt  s1, t5, child_loop   # repeat until all 8 children done

    # mark parent as internal now
    slli t0, s0, 2
    la   t1, node_is_leaf
    add  t1, t1, t0
    sw   x0, 0(t1)            # node_is_leaf[parent] = 0 (not a leaf anymore)

    la   t1, node_particle
    add  t1, t1, t0
    li   t2, -1
    sw   t2, 0(t1)            # node_particle[parent] = -1

    ld   ra, 24(sp)
    ld   s0, 16(sp)
    ld   s1,  8(sp)
    ld   s2,  0(sp)
    addi sp, sp, 32
    ret


# insert_particle
# inserts body a0 into the octree starting at node a1
#
# there are 3 cases:
#   case 1: node is internal -> find correct octant child and recurse
#   case 2: node is empty leaf -> place body here directly
#   case 3: node is occupied leaf -> subdivide it,
#           re-insert old body, then insert new body
#
# arguments:
#   a0 = body index (0 to N-1)
#   a1 = node index to insert into
#
# saved registers:
#   s0 = body index
#   s1 = node index
#   s2 = octant result
#   s3 = unused but frame reserved
#   fs0 = unused but frame reserved


.globl insert_particle
insert_particle:
    addi sp, sp, -48
    sd   ra, 40(sp)
    sd   s0, 32(sp)
    sd   s1, 24(sp)
    sd   s2, 16(sp)
    sd   s3,  8(sp)
    fsd  fs0,  0(sp)

    mv   s0, a0               # s0 = body to insert
    mv   s1, a1               # s1 = target node

    slli t0, s1, 2            # t0 = node_index * 4

    # check if node is a leaf or internal
    la   t1, node_is_leaf
    add  t1, t1, t0
    lw   t2, 0(t1)            # t2 = is_leaf flag

    la   t1, node_particle
    add  t1, t1, t0
    lw   t3, 0(t1)            # t3 = existing body index (-1 if empty)

    beqz t2, internal_node    # is_leaf == 0 means internal, go recurse

    # node is a leaf, check if empty
    li   t4, -1
    beq  t3, t4, empty_leaf   # particle == -1 means empty, place body here

    # occupied leaf: subdivide first then re-insert both bodies
    mv   a0, s1
    call subdivide            # split this node into 8 children

    mv   a0, t3               # a0 = old body that was here
    mv   a1, s1               # a1 = this node (now internal)
    call insert_particle      # put old body into correct child

    mv   a0, s0               # a0 = new body
    mv   a1, s1
    call insert_particle      # put new body into correct child
    j    insert_done

empty_leaf:
    # place body s0 directly in this empty leaf
    slli t2, s0, 2            # t2 = body_index * 4

    la   t1, node_particle
    add  t1, t1, t0
    sw   s0, 0(t1)            # node_particle[node] = body index

    # copy mass of this body into the node
    la   t1, mass
    add  t1, t1, t2
    flw  ft0, 0(t1)           # ft0 = mass[s0]
    la   t1, node_mass
    add  t1, t1, t0
    fsw  ft0, 0(t1)           # node_mass[node] = mass[s0]

    # copy position as centre of mass (single body so com = position)
    la   t1, px
    add  t1, t1, t2
    flw  ft1, 0(t1)           # ft1 = px[s0]
    la   t1, node_com_x
    add  t1, t1, t0
    fsw  ft1, 0(t1)           # node_com_x[node] = px[s0]

    la   t1, py
    add  t1, t1, t2
    flw  ft2, 0(t1)           # ft2 = py[s0]
    la   t1, node_com_y
    add  t1, t1, t0
    fsw  ft2, 0(t1)           # node_com_y[node] = py[s0]

    la   t1, pz
    add  t1, t1, t2
    flw  ft3, 0(t1)           # ft3 = pz[s0]
    la   t1, node_com_z
    add  t1, t1, t0
    fsw  ft3, 0(t1)           # node_com_z[node] = pz[s0]
    j    insert_done

internal_node:
    # node is internal, find which child octant this body belongs in
    slli t2, s0, 2            # t2 = body_index * 4

    la   t1, px
    add  t1, t1, t2
    flw  fa0, 0(t1)           # fa0 = px[s0] (float arg for get_octant)

    la   t1, py
    add  t1, t1, t2
    flw  fa1, 0(t1)           # fa1 = py[s0]

    la   t1, pz
    add  t1, t1, t2
    flw  fa2, 0(t1)           # fa2 = pz[s0]

    mv   a0, s1               # a0 = node index
    call get_octant           # returns octant 0-7 in a0
    mv   s2, a0               # s2 = octant

    # compute byte offset into node_child[s1*8 + octant]
    slli t0, s1, 3            # t0 = node_index * 8
    add  t0, t0, s2           # t0 = node_index*8 + octant
    slli t0, t0, 2            # t0 *= 4 (byte offset)
    la   t1, node_child
    add  t1, t1, t0
    lw   a1, 0(t1)            # a1 = child node index

    mv   a0, s0               # a0 = body to insert
    call insert_particle      # recurse into the correct child

insert_done:
    fld  fs0,  0(sp)
    ld   s3,   8(sp)
    ld   s2,  16(sp)
    ld   s1,  24(sp)
    ld   s0,  32(sp)
    ld   ra,  40(sp)
    addi sp, sp, 48
    ret


# compute_mass
# walks the tree bottom-up computing total mass and centre of mass
# for each internal node by summing over children
#
# for leaf nodes: mass/com was already set when we inserted the body
# for internal nodes:
#   total_mass = sum of child masses
#   com_x = sum(child_mass * child_com_x) / total_mass
#   com_y and com_z same way
#
# argument: a0 = node index to start from (call with root)
# returns:  a0 = same node index (unchanged)
#
# saved registers:
#   s0 = node index
#   s1 = child loop counter 0 to 7
#
# float registers:
#   ft0 = running total_mass accumulator
#   ft1 = weighted sum of com_x
#   ft2 = weighted sum of com_y
#   ft3 = weighted sum of com_z
#   ft4 = mass of current child
#   ft5 = child com component then weighted product


.globl compute_mass
compute_mass:
    addi sp, sp, -48
    sd   ra, 40(sp)
    sd   s0, 32(sp)
    sd   s1, 24(sp)
    sd   s2, 16(sp)
    sd   s3,  8(sp)
    fsd  fs0,  0(sp)

    mv   s0, a0               # s0 = node index

    slli t0, s0, 2            # t0 = node_index * 4
    la   t1, node_is_leaf
    add  t1, t1, t0
    lw   t2, 0(t1)            # t2 = is_leaf
    bnez t2, leaf_mass        # leaf nodes already have mass set, skip

    # internal node: accumulate mass and com from all 8 children
    fcvt.s.w ft0, x0          # ft0 = 0.0 (total_mass starts at zero)
    fcvt.s.w ft1, x0          # ft1 = 0.0 (weighted com_x sum)
    fcvt.s.w ft2, x0          # ft2 = 0.0 (weighted com_y sum)
    fcvt.s.w ft3, x0          # ft3 = 0.0 (weighted com_z sum)

    li   s1, 0                # s1 = child index starting at 0

child_sum_loop:
    # compute byte offset into node_child[s0*8 + s1]
    slli t1, s0, 3            # t1 = s0 * 8
    add  t1, t1, s1           # t1 = s0*8 + s1
    slli t1, t1, 2            # t1 *= 4 (byte offset)
    la   t2, node_child
    add  t2, t2, t1
    lw   a0, 0(t2)            # a0 = child node index
    bltz a0, next_child       # child == -1 means no child here, skip

    call compute_mass         # recurse into child first (post-order)
                              # after return a0 still holds child index

    slli t1, a0, 2            # t1 = child_index * 4

    la   t2, node_mass
    add  t2, t2, t1
    flw  ft4, 0(t2)           # ft4 = child_mass

    fadd.s ft0, ft0, ft4      # total_mass += child_mass

    la   t2, node_com_x
    add  t2, t2, t1
    flw  ft5, 0(t2)           # ft5 = child_com_x
    fmul.s ft5, ft5, ft4      # ft5 = child_com_x * child_mass
    fadd.s ft1, ft1, ft5      # weighted_com_x += child_com_x * child_mass

    la   t2, node_com_y
    add  t2, t2, t1
    flw  ft5, 0(t2)           # ft5 = child_com_y
    fmul.s ft5, ft5, ft4      # ft5 = child_com_y * child_mass
    fadd.s ft2, ft2, ft5      # weighted_com_y += child_com_y * child_mass

    la   t2, node_com_z
    add  t2, t2, t1
    flw  ft5, 0(t2)           # ft5 = child_com_z
    fmul.s ft5, ft5, ft4      # ft5 = child_com_z * child_mass
    fadd.s ft3, ft3, ft5      # weighted_com_z += child_com_z * child_mass

next_child:
    addi s1, s1, 1            # s1++ next child
    li   t1, 8
    blt  s1, t1, child_sum_loop  # loop until all 8 children done

    # divide weighted sums by total mass to get true centre of mass
    slli t0, s0, 2

    la   t1, node_mass
    add  t1, t1, t0
    fsw  ft0, 0(t1)           # node_mass[node] = total_mass

    fdiv.s ft1, ft1, ft0      # ft1 = weighted_com_x / total_mass = com_x
    fdiv.s ft2, ft2, ft0      # ft2 = com_y
    fdiv.s ft3, ft3, ft0      # ft3 = com_z

    la   t1, node_com_x
    add  t1, t1, t0
    fsw  ft1, 0(t1)           # node_com_x[node] = com_x

    la   t1, node_com_y
    add  t1, t1, t0
    fsw  ft2, 0(t1)           # node_com_y[node] = com_y

    la   t1, node_com_z
    add  t1, t1, t0
    fsw  ft3, 0(t1)           # node_com_z[node] = com_z
    j    mass_done

leaf_mass:
    nop                       # nothing to do, mass/com already set on insert

mass_done:
    mv   a0, s0               # return node index unchanged
    fld  fs0,  0(sp)
    ld   s3,   8(sp)
    ld   s2,  16(sp)
    ld   s1,  24(sp)
    ld   s0,  32(sp)
    ld   ra,  40(sp)
    addi sp, sp, 48
    ret


# calculate_force_bh
# computes the gravitational force on one body from the entire tree
# uses the barnes-hut criterion to decide whether to approximate
# a group of bodies as a single point mass or recurse deeper
#
# the criterion: if node_size / dist < theta then treat as point mass
# otherwise recurse into children
#
# arguments:
#   a0  = body index i
#   a1  = node index to evaluate
#   fa0 = theta (opening angle)
#
# saved registers:
#   s0  = body index i
#   s1  = node index
#   s2  = child loop counter
#   fs0 = theta (must survive recursive calls)
#
# float registers:
#   ft0  = node_mass
#   ft3  = dx = node_com_x - px[i]  (kept until force applied)
#   ft6  = dy = node_com_y - py[i]  (kept)
#   ft9  = dz = node_com_z - pz[i]  (kept)
#   ft10 = dist = sqrt(dx^2+dy^2+dz^2)
#   ft11 = node_size / dist (opening angle ratio)
#   in open_node section:
#   ft4  = softening^2 then intermediate denom
#   ft5  = dist^2 + soft^2 then r^3 (the softened denominator)
#   ft7  = G then G*mass
#   ft10 = acc_factor = G*mass/denom
#   ft8  = acc_factor * direction component
#   ft11 = existing acceleration component then updated value


.globl calculate_force_bh
calculate_force_bh:
    addi sp, sp, -64
    sd   ra, 56(sp)
    sd   s0, 48(sp)
    sd   s1, 40(sp)
    sd   s2, 32(sp)
    sd   s3, 24(sp)
    fsd  fs0, 16(sp)          # fs0 is callee-saved float, must save/restore
    fsd  fs1,  8(sp)
    fsd  fs2,  0(sp)

    mv   s0, a0               # s0 = body index i
    mv   s1, a1               # s1 = node index
    fmv.s fs0, fa0            # fs0 = theta, kept across recursive calls

    slli t0, s1, 2            # t0 = node_index * 4

    # skip this node if it has zero mass
    la   t1, node_mass
    add  t1, t1, t0
    flw  ft0, 0(t1)           # ft0 = node_mass

    fcvt.s.w ft4, x0          # ft4 = 0.0 (need float zero for comparison)
    feq.s t2, ft0, ft4        # t2 = 1 if node_mass == 0.0
    bnez t2, force_done       # zero mass node contributes nothing

    # if this is a leaf containing the same body, skip (no self-force)
    la   t1, node_is_leaf
    add  t1, t1, t0
    lw   t2, 0(t1)            # t2 = is_leaf

    beqz t2, not_self_check   # internal node cant be self, skip check

    la   t1, node_particle
    add  t1, t1, t0
    lw   t2, 0(t1)            # t2 = particle index stored in leaf
    beq  t2, s0, force_done   # same body, skip self-force

not_self_check:
    # compute vector from body i to node centre of mass
    slli t3, s0, 2            # t3 = body_index * 4

    la   t1, px
    add  t1, t1, t3
    flw  ft1, 0(t1)           # ft1 = px[i]
    la   t1, node_com_x
    add  t1, t1, t0
    flw  ft2, 0(t1)           # ft2 = node_com_x
    fsub.s ft3, ft2, ft1      # ft3 = dx = node_com_x - px[i]

    la   t1, py
    add  t1, t1, t3
    flw  ft4, 0(t1)           # ft4 = py[i]
    la   t1, node_com_y
    add  t1, t1, t0
    flw  ft5, 0(t1)           # ft5 = node_com_y
    fsub.s ft6, ft5, ft4      # ft6 = dy = node_com_y - py[i]

    la   t1, pz
    add  t1, t1, t3
    flw  ft7, 0(t1)           # ft7 = pz[i]
    la   t1, node_com_z
    add  t1, t1, t0
    flw  ft8, 0(t1)           # ft8 = node_com_z
    fsub.s ft9, ft8, ft7      # ft9 = dz = node_com_z - pz[i]

    # compute distance = sqrt(dx^2 + dy^2 + dz^2)
    fmul.s ft10, ft3, ft3     # ft10 = dx * dx
    fmul.s ft11, ft6, ft6     # ft11 = dy * dy
    fadd.s ft10, ft10, ft11   # ft10 = dx^2 + dy^2
    fmul.s ft11, ft9, ft9     # ft11 = dz * dz
    fadd.s ft10, ft10, ft11   # ft10 = dx^2 + dy^2 + dz^2 = dist_sq
    fsqrt.s ft10, ft10        # ft10 = dist = sqrt(dist_sq)

    # compute barnes-hut ratio: s/d = node_size / dist
    la   t1, node_size
    add  t1, t1, t0
    flw  ft11, 0(t1)          # ft11 = node_size (s)
    fdiv.s ft11, ft11, ft10   # ft11 = s/d

    # if s/d < theta then node is far enough, treat as point mass
    flt.s t2, ft11, fs0       # t2 = 1 if (s/d) < theta
    bnez t2, open_node        # yes, apply force from this whole node

    # s/d >= theta, need to go deeper
    # if its a leaf (and not self, already checked), apply force directly
    la   t1, node_is_leaf
    add  t1, t1, t0
    lw   t2, 0(t1)
    bnez t2, open_node        # leaf: apply force now

    # internal and too close: recurse into all 8 children
    li   s2, 0                # s2 = child counter

recurse_loop:
    slli t1, s1, 3            # t1 = node_index * 8
    add  t1, t1, s2           # t1 = node_index*8 + child_i
    slli t1, t1, 2            # t1 *= 4 (byte offset)
    la   t2, node_child
    add  t2, t2, t1
    lw   a1, 0(t2)            # a1 = child node index
    bltz a1, recurse_next     # -1 means no child here

    mv   a0, s0               # a0 = body index
    fmv.s fa0, fs0            # fa0 = theta (pass to recursive call)
    call calculate_force_bh   # recurse into this child

recurse_next:
    addi s2, s2, 1
    li   t1, 8
    blt  s2, t1, recurse_loop # loop through all 8 children
    j    force_done

open_node:
    # apply gravitational acceleration from this node treated as a point mass
    #
    # formula:
    #   softened_r_sq = dist^2 + softening^2
    #   denom = softened_r_sq^(3/2) = softened_r_sq * sqrt(softened_r_sq)
    #   acc_factor = G * node_mass / denom
    #   ax[i] += acc_factor * dx  (and same for y and z)
    #
    # ft3=dx, ft6=dy, ft9=dz still valid from above
    # ft0=node_mass still valid from above
    # ft10=dist still valid from above

    la   t1, softening_float
    flw  ft4, 0(t1)           # ft4 = softening length

    fmul.s ft4, ft4, ft4      # ft4 = softening^2

    fmul.s ft5, ft10, ft10    # ft5 = dist^2 (recompute from ft10=dist)
    fadd.s ft5, ft5, ft4      # ft5 = dist^2 + soft^2 = softened_dist_sq

    fsqrt.s ft4, ft5          # ft4 = sqrt(softened_dist_sq)
    fmul.s  ft5, ft5, ft4     # ft5 = softened_dist_sq * sqrt(...) = r^3 (denom)

    la   t1, G_float
    flw  ft7, 0(t1)           # ft7 = G

    fmul.s ft7, ft7, ft0      # ft7 = G * node_mass
    fdiv.s ft10, ft7, ft5     # ft10 = acc_factor = G * node_mass / r^3

    slli t3, s0, 2            # t3 = body_index * 4

    # ax[i] += acc_factor * dx
    la   t1, ax
    add  t1, t1, t3
    flw  ft11, 0(t1)          # ft11 = current ax[i]
    fmul.s ft8, ft10, ft3     # ft8  = acc_factor * dx
    fadd.s ft11, ft11, ft8    # ft11 = ax[i] + acc_factor*dx
    fsw  ft11, 0(t1)          # write updated ax[i]

    # ay[i] += acc_factor * dy
    la   t1, ay
    add  t1, t1, t3
    flw  ft11, 0(t1)          # ft11 = current ay[i]
    fmul.s ft8, ft10, ft6     # ft8  = acc_factor * dy
    fadd.s ft11, ft11, ft8    # ft11 = ay[i] + acc_factor*dy
    fsw  ft11, 0(t1)          # write updated ay[i]

    # az[i] += acc_factor * dz
    la   t1, az
    add  t1, t1, t3
    flw  ft11, 0(t1)          # ft11 = current az[i]
    fmul.s ft8, ft10, ft9     # ft8  = acc_factor * dz
    fadd.s ft11, ft11, ft8    # ft11 = az[i] + acc_factor*dz
    fsw  ft11, 0(t1)          # write updated az[i]

force_done:
    fld  fs2,  0(sp)
    fld  fs1,  8(sp)
    fld  fs0, 16(sp)
    ld   s3,  24(sp)
    ld   s2,  32(sp)
    ld   s1,  40(sp)
    ld   s0,  48(sp)
    ld   ra,  56(sp)
    addi sp, sp, 64
    ret


# compute_bounding_box
# finds the min/max positions across all bodies to build a bounding box
# then allocates the root node and sets its geometry
# saved registers:
#   s0 = root node index after allocation
#
# float registers:
#   ft0 = min_x (running minimum)
#   ft1 = max_x (running maximum)
#   ft2 = min_y
#   ft3 = max_y
#   ft4 = min_z
#   ft5 = max_z
#   ft6 = current particle value being compared
#   ft7 = 2.0 constant
#   ft8 = centre x
#   ft9 = centre y
#   ft10 = centre z
#   ft11 = max side length then final_size with padding

.globl compute_bounding_box
compute_bounding_box:
    addi sp, sp, -16
    sd   ra,  8(sp)
    sd   s0,  0(sp)

    # initialise min/max with first body position
    la   t0, px
    flw  ft0, 0(t0)           # ft0 = min_x = px[0]
    flw  ft1, 0(t0)           # ft1 = max_x = px[0]

    la   t1, py
    flw  ft2, 0(t1)           # ft2 = min_y = py[0]
    flw  ft3, 0(t1)           # ft3 = max_y = py[0]

    la   t2, pz
    flw  ft4, 0(t2)           # ft4 = min_z = pz[0]
    flw  ft5, 0(t2)           # ft5 = max_z = pz[0]

    li   s0, 1                # loop counter i = 1
    li   t6, N
    beq  s0, t6, skip_loop   # if only one body skip loop

bbox_loop:
    slli t3, s0, 2            # t3 = i * 4

    la   t4, px
    add  t4, t4, t3
    flw  ft6, 0(t4)           # ft6 = px[i]
    fmin.s ft0, ft0, ft6      # update min_x
    fmax.s ft1, ft1, ft6      # update max_x

    la   t4, py
    add  t4, t4, t3
    flw  ft6, 0(t4)           # ft6 = py[i]
    fmin.s ft2, ft2, ft6      # update min_y
    fmax.s ft3, ft3, ft6      # update max_y

    la   t4, pz
    add  t4, t4, t3
    flw  ft6, 0(t4)           # ft6 = pz[i]
    fmin.s ft4, ft4, ft6      # update min_z
    fmax.s ft5, ft5, ft6      # update max_z

    addi s0, s0, 1
    blt  s0, t6, bbox_loop    # loop until all bodies checked

skip_loop:
    # compute the cube side length with 2% padding to avoid edge cases
    li   t0, 2
    fcvt.s.w ft7, t0          # ft7 = 2.0

    fsub.s ft6, ft1, ft0      # ft6 = width_x = max_x - min_x
    fsub.s ft9, ft3, ft2      # ft9 = width_y
    fsub.s ft10, ft5, ft4     # ft10 = width_z

    fmax.s ft11, ft6, ft9     # ft11 = max(width_x, width_y)
    fmax.s ft11, ft11, ft10   # ft11 = largest span across all axes

    li   t0, 100
    fcvt.s.w ft6, t0          # ft6 = 100.0
    fdiv.s ft6, ft11, ft6     # ft6 = max_size / 100 = 1% padding amount
    fadd.s ft11, ft11, ft6    # add 1%
    fadd.s ft11, ft11, ft6    # add another 1% -> total 2% padding

    # compute root centre = midpoint of bounding box
    fadd.s ft8, ft0, ft1      # ft8 = min_x + max_x
    fdiv.s ft8, ft8, ft7      # ft8 = cx = (min_x + max_x) / 2

    fadd.s ft9, ft2, ft3      # ft9 = min_y + max_y
    fdiv.s ft9, ft9, ft7      # ft9 = cy

    fadd.s ft10, ft4, ft5     # ft10 = min_z + max_z
    fdiv.s ft10, ft10, ft7    # ft10 = cz

    # allocate the root node
    call allocate_node        # a0 = root node index
    mv   s0, a0               # s0 = root

    slli t0, s0, 2            # t0 = root_index * 4

    # store root centre
    la   t1, node_cx
    add  t1, t1, t0
    fsw  ft8, 0(t1)           # node_cx[root] = cx

    la   t1, node_cy
    add  t1, t1, t0
    fsw  ft9, 0(t1)           # node_cy[root] = cy

    la   t1, node_cz
    add  t1, t1, t0
    fsw  ft10, 0(t1)          # node_cz[root] = cz

    # store root size
    la   t1, node_size
    add  t1, t1, t0
    fsw  ft11, 0(t1)          # node_size[root] = final_size

    # compute and store root bounding box
    fdiv.s ft6, ft11, ft7     # ft6 = half_size = final_size / 2

    fsub.s ft0, ft8, ft6      # xmin = cx - half_size
    fadd.s ft1, ft8, ft6      # xmax = cx + half_size
    la   t1, node_xmin
    add  t1, t1, t0
    fsw  ft0, 0(t1)
    la   t1, node_xmax
    add  t1, t1, t0
    fsw  ft1, 0(t1)

    fsub.s ft0, ft9, ft6      # ymin = cy - half_size
    fadd.s ft1, ft9, ft6      # ymax = cy + half_size
    la   t1, node_ymin
    add  t1, t1, t0
    fsw  ft0, 0(t1)
    la   t1, node_ymax
    add  t1, t1, t0
    fsw  ft1, 0(t1)

    fsub.s ft0, ft10, ft6     # zmin = cz - half_size
    fadd.s ft1, ft10, ft6     # zmax = cz + half_size
    la   t1, node_zmin
    add  t1, t1, t0
    fsw  ft0, 0(t1)
    la   t1, node_zmax
    add  t1, t1, t0
    fsw  ft1, 0(t1)

    # mark root as empty leaf to start with
    la   t1, node_is_leaf
    add  t1, t1, t0
    li   t2, 1
    sw   t2, 0(t1)            # node_is_leaf[root] = 1

    la   t1, node_particle
    add  t1, t1, t0
    li   t2, -1
    sw   t2, 0(t1)            # node_particle[root] = -1

    # zero root mass and com
    fcvt.s.w ft0, x0          # ft0 = 0.0
    la   t1, node_mass
    add  t1, t1, t0
    fsw  ft0, 0(t1)
    la   t1, node_com_x
    add  t1, t1, t0
    fsw  ft0, 0(t1)
    la   t1, node_com_y
    add  t1, t1, t0
    fsw  ft0, 0(t1)
    la   t1, node_com_z
    add  t1, t1, t0
    fsw  ft0, 0(t1)

    mv   a0, s0               # return root node index
    ld   ra,  8(sp)
    ld   s0,  0(sp)
    addi sp, sp, 16
    ret


# build_tree
# rebuilds the entire octree from scratch each timestep

# steps:
#   1. reset node_count to 0
#   2. clear node_child array to all -1
#   3. reset node_is_leaf/particle/mass for all nodes (fix stale state bug)
#   4. compute bounding box and allocate root
#   5. insert all N bodies
#   6. compute mass and com bottom-up

# returns: a0 = root node index
# saved registers:
#   s0 = root node index
#   s1 = loop counter for inserting bodies

.globl build_tree
build_tree:
    addi sp, sp, -32
    sd   ra, 24(sp)
    sd   s0, 16(sp)
    sd   s1,  8(sp)

    # reset node count to 0 so we reuse the pool from the start
    la   t0, node_count
    sw   x0, 0(t0)            # node_count = 0

    # clear all node_child entries to -1
    # total slots = MAX_NODES * 8, each is 4 bytes
    li   t0, 0
    li   t1, MAX_NODES
    slli t1, t1, 3            # t1 = MAX_NODES * 8 (total child slots)
    la   t2, node_child       # t2 = base of child array

clear_loop:
    li   t3, -1
    sw   t3, 0(t2)            # node_child[t0] = -1
    addi t2, t2, 4            # advance to next slot
    addi t0, t0, 1
    blt  t0, t1, clear_loop   # repeat until all cleared

    # reset node_is_leaf, node_particle, and node_mass for all MAX_NODES
    # this is critical because nodes from the previous step are reused
    # without this, stale is_leaf=0 values cause insert_particle to go wrong
    li   t0, 0
    li   t1, MAX_NODES

reset_nodes_loop:
    slli t3, t0, 2            # t3 = i * 4

    la   t4, node_is_leaf
    add  t4, t4, t3
    li   t5, 1
    sw   t5, 0(t4)            # node_is_leaf[i] = 1 (fresh empty leaf)

    la   t4, node_particle
    add  t4, t4, t3
    li   t5, -1
    sw   t5, 0(t4)            # node_particle[i] = -1 (no body)

    la   t4, node_mass
    add  t4, t4, t3
    sw   x0, 0(t4)            # node_mass[i] = 0 (zero mass)

    addi t0, t0, 1
    blt  t0, t1, reset_nodes_loop  # loop until all MAX_NODES reset

    # compute bounding box and allocate root node
    call compute_bounding_box # a0 = root index
    mv   s0, a0               # s0 = root

    # insert all N bodies into the tree one by one
    li   s1, 0

insert_all:
    mv   a0, s1               # a0 = body index
    mv   a1, s0               # a1 = root node
    call insert_particle      # insert body s1 into tree
    addi s1, s1, 1
    li   t0, N
    blt  s1, t0, insert_all   # loop until all bodies inserted

    # compute mass and centre of mass for all internal nodes
    mv   a0, s0
    call compute_mass         # propagates mass/com up from leaves

    mv   a0, s0               # return root index
    ld   ra, 24(sp)
    ld   s0, 16(sp)
    ld   s1,  8(sp)
    addi sp, sp, 32
    ret




# calculate_forces_bh
# zeroes all accelerations first then computes barnes-hut force for each body

# saved registers:
#   s0 = root node index
#   s1 = loop counter i
#
# float registers:
#   ft0 = 0.0 for zeroing the acceleration arrays



.globl calculate_forces_bh
calculate_forces_bh:
    addi sp, sp, -32
    sd   ra, 24(sp)
    sd   s0, 16(sp)
    sd   s1,  8(sp)

    mv   s0, a0               # s0 = root node index

    # zero out all acceleration arrays before computing new forces
    fcvt.s.w ft0, x0          # ft0 = 0.0
    li   s1, 0

zero_loop:
    slli t0, s1, 2            # t0 = i * 4

    la   t1, ax
    add  t1, t1, t0
    fsw  ft0, 0(t1)           # ax[i] = 0.0

    la   t1, ay
    add  t1, t1, t0
    fsw  ft0, 0(t1)           # ay[i] = 0.0

    la   t1, az
    add  t1, t1, t0
    fsw  ft0, 0(t1)           # az[i] = 0.0

    addi s1, s1, 1
    li   t0, N
    blt  s1, t0, zero_loop    # loop until all zeroed

    # compute forces for each body using the tree
    li   s1, 0

force_loop:
    mv   a0, s1               # a0 = body index i
    mv   a1, s0               # a1 = root node

    la   t0, theta_float
    flw  fa0, 0(t0)           # fa0 = theta (opening angle)

    call calculate_force_bh   # adds force contribution to ax/ay/az for body i

    addi s1, s1, 1
    li   t0, N
    blt  s1, t0, force_loop   # loop until all bodies done

    ld   ra, 24(sp)
    ld   s0, 16(sp)
    ld   s1,  8(sp)
    addi sp, sp, 32
    ret


# calculate_kick
# leapfrog half-kick: v[i] += a[i] * half_dt for all bodies
# this is called twice per timestep (before and after drift)
#
# registers used:
#   t0-t5 = base addresses of vx,vy,vz,ax,ay,az arrays
#   t6    = scratch address for element access
#   s0    = loop counter i
#   s1    = N (loop limit)
#   s2    = byte offset = i * 4
#   ft0   = half_dt (loaded once, reused every iteration)
#   ft1   = velocity component being updated
#   ft2   = acceleration component
#   ft3   = a[i] * half_dt (the increment to add)


.globl calculate_kick
calculate_kick:
    addi sp, sp, -32
    sd   ra, 24(sp)
    sd   s0, 16(sp)
    sd   s1,  8(sp)
    sd   s2,  0(sp)

    la   t0, vx               # t0 = base of vx
    la   t1, vy               # t1 = base of vy
    la   t2, vz               # t2 = base of vz
    la   t3, ax               # t3 = base of ax
    la   t4, ay               # t4 = base of ay
    la   t5, az               # t5 = base of az

    la   t6, half_dt_float
    flw  ft0, 0(t6)           # ft0 = half_dt (constant for all iterations)

    li   s0, 0                # i = 0
    li   s1, N                # loop limit

kick_loop:
    beq  s0, s1, kick_end     # if i == N we are done

    slli s2, s0, 2            # s2 = i * 4 (byte offset)

    # vx[i] += ax[i] * half_dt
    add  t6, t0, s2
    flw  ft1, 0(t6)           # ft1 = vx[i]
    add  t6, t3, s2
    flw  ft2, 0(t6)           # ft2 = ax[i]
    fmul.s ft3, ft2, ft0      # ft3 = ax[i] * half_dt
    fadd.s ft1, ft1, ft3      # ft1 = vx[i] + ax[i]*half_dt
    add  t6, t0, s2
    fsw  ft1, 0(t6)           # store updated vx[i]

    # vy[i] += ay[i] * half_dt
    add  t6, t1, s2
    flw  ft1, 0(t6)           # ft1 = vy[i]
    add  t6, t4, s2
    flw  ft2, 0(t6)           # ft2 = ay[i]
    fmul.s ft3, ft2, ft0      # ft3 = ay[i] * half_dt
    fadd.s ft1, ft1, ft3      # ft1 = vy[i] + ay[i]*half_dt
    add  t6, t1, s2
    fsw  ft1, 0(t6)           # store updated vy[i]

    # vz[i] += az[i] * half_dt
    add  t6, t2, s2
    flw  ft1, 0(t6)           # ft1 = vz[i]
    add  t6, t5, s2
    flw  ft2, 0(t6)           # ft2 = az[i]
    fmul.s ft3, ft2, ft0      # ft3 = az[i] * half_dt
    fadd.s ft1, ft1, ft3      # ft1 = vz[i] + az[i]*half_dt
    add  t6, t2, s2
    fsw  ft1, 0(t6)           # store updated vz[i]

    addi s0, s0, 1            # i++
    j    kick_loop

kick_end:
    ld   ra, 24(sp)
    ld   s0, 16(sp)
    ld   s1,  8(sp)
    ld   s2,  0(sp)
    addi sp, sp, 32
    ret


# calculate_drift
# leapfrog drift: p[i] += v[i] * dt for all bodies
# this moves positions forward using current velocities
#
# registers used:
#   t0-t5 = base addresses of px,py,pz,vx,vy,vz
#   t6    = scratch address
#   s0    = loop counter i
#   s1    = N
#   s2    = byte offset = i * 4
#   ft0   = dt (full timestep, loaded once)
#   ft1   = position component being updated
#   ft2   = velocity component
#   ft3   = v[i] * dt (the displacement to add)



.globl calculate_drift
calculate_drift:
    addi sp, sp, -32
    sd   ra, 24(sp)
    sd   s0, 16(sp)
    sd   s1,  8(sp)
    sd   s2,  0(sp)

    la   t0, px
    la   t1, py
    la   t2, pz
    la   t3, vx
    la   t4, vy
    la   t5, vz

    la   t6, dt_float
    flw  ft0, 0(t6)           # ft0 = dt (full timestep)

    li   s0, 0
    li   s1, N

drift_loop:
    beq  s0, s1, drift_end

    slli s2, s0, 2            # s2 = i * 4

    # px[i] += vx[i] * dt
    add  t6, t0, s2
    flw  ft1, 0(t6)           # ft1 = px[i]
    add  t6, t3, s2
    flw  ft2, 0(t6)           # ft2 = vx[i]
    fmul.s ft3, ft2, ft0      # ft3 = vx[i] * dt
    fadd.s ft1, ft1, ft3      # ft1 = px[i] + vx[i]*dt
    add  t6, t0, s2
    fsw  ft1, 0(t6)           # store updated px[i]

    # py[i] += vy[i] * dt
    add  t6, t1, s2
    flw  ft1, 0(t6)           # ft1 = py[i]
    add  t6, t4, s2
    flw  ft2, 0(t6)           # ft2 = vy[i]
    fmul.s ft3, ft2, ft0      # ft3 = vy[i] * dt
    fadd.s ft1, ft1, ft3
    add  t6, t1, s2
    fsw  ft1, 0(t6)           # store updated py[i]

    # pz[i] += vz[i] * dt
    add  t6, t2, s2
    flw  ft1, 0(t6)           # ft1 = pz[i]
    add  t6, t5, s2
    flw  ft2, 0(t6)           # ft2 = vz[i]
    fmul.s ft3, ft2, ft0      # ft3 = vz[i] * dt
    fadd.s ft1, ft1, ft3
    add  t6, t2, s2
    fsw  ft1, 0(t6)           # store updated pz[i]

    addi s0, s0, 1
    j    drift_loop

drift_end:
    ld   ra, 24(sp)
    ld   s0, 16(sp)
    ld   s1,  8(sp)
    ld   s2,  0(sp)
    addi sp, sp, 32
    ret
