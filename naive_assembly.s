
.section .text

# calculate_kick: v[i] += a[i] * half_dt
# Uses: t0-t5 (base addrs), t6 (scratch)
#       s0 (i), s1 (n=100), s2 (byte offset i*4)
#       ft0 (half_dt), ft1-ft3 (temps)

.globl calculate_kick
calculate_kick:
    addi sp, sp, -32
    sd ra, 24(sp)
    sd s0, 16(sp)
    sd s1,  8(sp)
    sd s2,  0(sp)

    la t0, vx
    la t1, vy
    la t2, vz
    la t3, ax
    la t4, ay
    la t5, az

    la t6, half_dt_float
    flw ft0, 0(t6)          # ft0 = half_dt

    addi s0, x0, 0
    li s1, 100              # n = 100

kick_loop:
    beq s0, s1, kick_end

    slli s2, s0, 2          # byte offset = i * 4

    # vx[i] += ax[i] * half_dt
    add t6, t0, s2
    flw ft1, 0(t6)          # ft1 = vx[i]
    add t6, t3, s2
    flw ft2, 0(t6)          # ft2 = ax[i]
    fmul.s ft3, ft2, ft0    # ft3 = ax[i] * half_dt
    fadd.s ft1, ft1, ft3    # ft1 = vx[i] + ax[i]*half_dt
    add t6, t0, s2
    fsw ft1, 0(t6)          # store vx[i]

    # vy[i] += ay[i] * half_dt
    add t6, t1, s2
    flw ft1, 0(t6)          # ft1 = vy[i]
    add t6, t4, s2
    flw ft2, 0(t6)          # ft2 = ay[i]
    fmul.s ft3, ft2, ft0
    fadd.s ft1, ft1, ft3
    add t6, t1, s2
    fsw ft1, 0(t6)

    # vz[i] += az[i] * half_dt
    add t6, t2, s2
    flw ft1, 0(t6)          # ft1 = vz[i]
    add t6, t5, s2
    flw ft2, 0(t6)          # ft2 = az[i]
    fmul.s ft3, ft2, ft0
    fadd.s ft1, ft1, ft3
    add t6, t2, s2
    fsw ft1, 0(t6)

    addi s0, s0, 1
    j kick_loop

kick_end:
    ld ra, 24(sp)
    ld s0, 16(sp)
    ld s1,  8(sp)
    ld s2,  0(sp)
    addi sp, sp, 32
    ret


# calculate_drift: p[i] += v[i] * dt
# Uses: t0-t5 (base addrs), t6 (scratch)
#       s0 (i), s1 (n), s2 (byte offset)
#       ft0 (dt), ft1-ft3 (temps)

.globl calculate_drift
calculate_drift:
    addi sp, sp, -32
    sd ra, 24(sp)
    sd s0, 16(sp)
    sd s1,  8(sp)
    sd s2,  0(sp)

    la t0, px
    la t1, py
    la t2, pz
    la t3, vx
    la t4, vy
    la t5, vz

    la t6, dt_float
    flw ft0, 0(t6)          # ft0 = dt

    addi s0, x0, 0
    li s1, 100              # n = 100

drift_loop:
    beq s0, s1, drift_end

    slli s2, s0, 2

    # px[i] += vx[i] * dt
    add t6, t0, s2
    flw ft1, 0(t6)          # ft1 = px[i]
    add t6, t3, s2
    flw ft2, 0(t6)          # ft2 = vx[i]
    fmul.s ft3, ft2, ft0
    fadd.s ft1, ft1, ft3
    add t6, t0, s2
    fsw ft1, 0(t6)

    # py[i] += vy[i] * dt
    add t6, t1, s2
    flw ft1, 0(t6)
    add t6, t4, s2
    flw ft2, 0(t6)
    fmul.s ft3, ft2, ft0
    fadd.s ft1, ft1, ft3
    add t6, t1, s2
    fsw ft1, 0(t6)

    # pz[i] += vz[i] * dt
    add t6, t2, s2
    flw ft1, 0(t6)
    add t6, t5, s2
    flw ft2, 0(t6)
    fmul.s ft3, ft2, ft0
    fadd.s ft1, ft1, ft3
    add t6, t2, s2
    fsw ft1, 0(t6)

    addi s0, s0, 1
    j drift_loop

drift_end:
    ld ra, 24(sp)
    ld s0, 16(sp)
    ld s1,  8(sp)
    ld s2,  0(sp)
    addi sp, sp, 32
    ret


# calculate_acc: compute gravitational acceleration for each body
#
# Outer loop i, inner loop j:
#   dx = px[j]-px[i], dy = py[j]-py[i], dz = pz[j]-pz[i]
#   dist_sq = dx^2 + dy^2 + dz^2
#   denom = (dist_sq + softening^2)^1.5   [approx: sqrt(dist_sq+s^2) * (dist_sq+s^2)]
#   acc_factor = G * mass[j] / denom
#   ax[i] += acc_factor * dx,  etc.
#
# Register map (staying within t0-t6, s0-s5):
#   t0=px, t1=py, t2=pz, t3=ax, t4=ay, t5=az, t6=mass (scratch reused)
#   s0=i, s1=n, s2=i*4, s3=j, s4=j*4, s5=scratch addr
#   ft0=G, ft1=softening, ft2-ft11=temps
#   (ft8 reused for softening^2, ft9 for denom)

.globl calculate_acc
calculate_acc:
    addi sp, sp, -64
    sd ra, 56(sp)
    sd s0, 48(sp)
    sd s1, 40(sp)
    sd s2, 32(sp)
    sd s3, 24(sp)
    sd s4, 16(sp)
    sd s5,  8(sp)

    la t0, px
    la t1, py
    la t2, pz
    la t3, ax
    la t4, ay
    la t5, az

    # Load G and softening into float regs
    la t6, G_float
    flw ft0, 0(t6)          # ft0 = G

    la t6, softening_float
    flw ft1, 0(t6)          # ft1 = softening

    addi s0, x0, 0
    li s1, 100              # n = 100

outer_loop:
    beq s0, s1, acc_end

    slli s2, s0, 2          # s2 = i*4

    # Zero out ax[i], ay[i], az[i]
    # Use fcvt.s.w to get 0.0 from integer zero
    fcvt.s.w ft2, x0        # ft2 = 0.0

    add s5, t3, s2
    fsw ft2, 0(s5)          # ax[i] = 0
    add s5, t4, s2
    fsw ft2, 0(s5)          # ay[i] = 0
    add s5, t5, s2
    fsw ft2, 0(s5)          # az[i] = 0

    # Load mass base — we need it in inner loop via t6
    # We'll reload t6 = mass each inner iteration (t6 is scratch)
    addi s3, x0, 0          # j = 0

inner_loop:
    beq s3, s1, inner_end
    beq s0, s3, skip_self   # skip i == j

    slli s4, s3, 2          # s4 = j*4

    # dx = px[j] - px[i]
    add s5, t0, s2
    flw ft2, 0(s5)          # ft2 = px[i]
    add s5, t0, s4
    flw ft3, 0(s5)          # ft3 = px[j]
    fsub.s ft4, ft3, ft2    # ft4 = dx

    # dy = py[j] - py[i]
    add s5, t1, s2
    flw ft2, 0(s5)          # ft2 = py[i]
    add s5, t1, s4
    flw ft3, 0(s5)          # ft3 = py[j]
    fsub.s ft5, ft3, ft2    # ft5 = dy

    # dz = pz[j] - pz[i]
    add s5, t2, s2
    flw ft2, 0(s5)          # ft2 = pz[i]
    add s5, t2, s4
    flw ft3, 0(s5)          # ft3 = pz[j]
    fsub.s ft6, ft3, ft2    # ft6 = dz

    # dist_sq = dx^2 + dy^2 + dz^2
    fmul.s ft7, ft4, ft4
    fmul.s ft8, ft5, ft5
    fmul.s ft9, ft6, ft6
    fadd.s ft7, ft7, ft8
    fadd.s ft7, ft7, ft9    # ft7 = dist_sq

    # denom approx = sqrt(dist_sq + soft^2) * (dist_sq + soft^2)
    fmul.s ft8, ft1, ft1    # ft8 = softening^2
    fadd.s ft7, ft7, ft8    # ft7 = dist_sq + soft^2
    fsqrt.s ft9, ft7        # ft9 = sqrt(dist_sq + soft^2)
    fmul.s ft9, ft9, ft7    # ft9 = denom ~ r^3

    # acc_factor = G * mass[j] / denom
    la t6, mass
    add s5, t6, s4
    flw ft10, 0(s5)         # ft10 = mass[j]
    fmul.s ft11, ft0, ft10  # ft11 = G * mass[j]
    fdiv.s ft11, ft11, ft9  # ft11 = acc_factor

    # ax[i] += acc_factor * dx
    add s5, t3, s2
    flw ft2, 0(s5)
    fmul.s ft3, ft11, ft4
    fadd.s ft2, ft2, ft3
    fsw ft2, 0(s5)

    # ay[i] += acc_factor * dy
    add s5, t4, s2
    flw ft2, 0(s5)
    fmul.s ft3, ft11, ft5
    fadd.s ft2, ft2, ft3
    fsw ft2, 0(s5)

    # az[i] += acc_factor * dz
    add s5, t5, s2
    flw ft2, 0(s5)
    fmul.s ft3, ft11, ft6
    fadd.s ft2, ft2, ft3
    fsw ft2, 0(s5)

skip_self:
    addi s3, s3, 1
    j inner_loop

inner_end:
    addi s0, s0, 1
    j outer_loop

acc_end:
    ld ra, 56(sp)
    ld s0, 48(sp)
    ld s1, 40(sp)
    ld s2, 32(sp)
    ld s3, 24(sp)
    ld s4, 16(sp)
    ld s5,  8(sp)
    addi sp, sp, 64
    ret
