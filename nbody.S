.section .data
.align 2
.include "mydata.S"        

.section .text              


.section .text

.globl calculate_kick
calculate_kick:
    addi sp, sp, -64
    sd ra, 56(sp)
    sd s0, 48(sp)
    sd s1, 40(sp)
    sd s2, 32(sp)
    sd s3, 24(sp)
    sd s4, 16(sp)
    sd s5,  8(sp)
    sd s6,  0(sp)

    la s0, vx
    la s1, vy
    la s2, vz
    la s3, ax
    la s4, ay
    la s5, az

    la t6, half_dt_float
    flw ft0, 0(t6)                    # ft0 = half_dt

    li s6, 100                        # s6 = N = remaining elements

kick_strip_loop:
    beqz s6, kick_done

    vsetvli t0, s6, e32, m1, ta, ma

    vle32.v v1, (s0)                  # v1 = vx[strip]
    vle32.v v2, (s1)                  # v2 = vy[strip]
    vle32.v v3, (s2)                  # v3 = vz[strip]
    vle32.v v4, (s3)                  # v4 = ax[strip]
    vle32.v v5, (s4)                  # v5 = ay[strip]
    vle32.v v6, (s5)                  # v6 = az[strip]

    vfmacc.vf v1, ft0, v4            # vx += half_dt * ax
    vfmacc.vf v2, ft0, v5            # vy += half_dt * ay
    vfmacc.vf v3, ft0, v6            # vz += half_dt * az

    vse32.v v1, (s0)
    vse32.v v2, (s1)
    vse32.v v3, (s2)

    slli t1, t0, 2                    # byte stride = vl * 4
    add  s0, s0, t1                   # advance all pointers
    add  s1, s1, t1
    add  s2, s2, t1
    add  s3, s3, t1
    add  s4, s4, t1
    add  s5, s5, t1
    sub  s6, s6, t0                   # remaining -= vl
    j kick_strip_loop

kick_done:
    ld s6,  0(sp)
    ld s5,  8(sp)
    ld s4, 16(sp)
    ld s3, 24(sp)
    ld s2, 32(sp)
    ld s1, 40(sp)
    ld s0, 48(sp)
    ld ra, 56(sp)
    addi sp, sp, 64
    ret


.globl calculate_drift
calculate_drift:
    addi sp, sp, -64
    sd ra, 56(sp)
    sd s0, 48(sp)
    sd s1, 40(sp)
    sd s2, 32(sp)
    sd s3, 24(sp)
    sd s4, 16(sp)
    sd s5,  8(sp)
    sd s6,  0(sp)

    la s0, px
    la s1, py
    la s2, pz
    la s3, vx
    la s4, vy
    la s5, vz

    la t6, dt_float
    flw ft0, 0(t6)                    # ft0 = dt

    li s6, 100

drift_strip_loop:
    beqz s6, drift_done
    vsetvli t0, s6, e32, m1, ta, ma

    vle32.v v1, (s0)                  # px[strip]
    vle32.v v2, (s1)                  # py[strip]
    vle32.v v3, (s2)                  # pz[strip]
    vle32.v v4, (s3)                  # vx[strip]
    vle32.v v5, (s4)                  # vy[strip]
    vle32.v v6, (s5)                  # vz[strip]

    vfmacc.vf v1, ft0, v4            # px += dt * vx
    vfmacc.vf v2, ft0, v5            # py += dt * vy
    vfmacc.vf v3, ft0, v6            # pz += dt * vz

    vse32.v v1, (s0)
    vse32.v v2, (s1)
    vse32.v v3, (s2)

    slli t1, t0, 2
    add  s0, s0, t1
    add  s1, s1, t1
    add  s2, s2, t1
    add  s3, s3, t1
    add  s4, s4, t1
    add  s5, s5, t1
    sub  s6, s6, t0
    j drift_strip_loop

drift_done:
    ld s6,  0(sp)
    ld s5,  8(sp)
    ld s4, 16(sp)
    ld s3, 24(sp)
    ld s2, 32(sp)
    ld s1, 40(sp)
    ld s0, 48(sp)
    ld ra, 56(sp)
    addi sp, sp, 64
    ret


.globl calculate_acc
calculate_acc:
    addi sp, sp, -112
    sd ra,  104(sp)
    sd s0,   96(sp)
    sd s1,   88(sp)
    sd s2,   80(sp)
    sd s3,   72(sp)
    sd s4,   64(sp)
    sd s5,   56(sp)
    sd s6,   48(sp)
    sd s7,   40(sp)
    sd s8,   32(sp)
    sd s9,   24(sp)
    sd s10,  16(sp)
    sd s11,   8(sp)

    la s4,  px
    la s5,  py
    la s6,  pz
    la s7,  ax
    la s8,  ay
    la s9,  az
    la s10, mass

    la t6, G_float
    flw ft0, 0(t6)                    # ft0 = G

    la t6, softening_float
    flw ft1, 0(t6)
    fmul.s ft1, ft1, ft1              

    fcvt.s.w ft2, x0                  # ft2 = 0.0 

    li s1, 100
    li s0, 0                          # i = 0

outer_i_loop:
    beq s0, s1, acc_done

    slli s11, s0, 2                   

  
    add t6, s4, s11
    flw ft3, 0(t6)                    # ft3 = px[i]
    add t6, s5, s11
    flw ft4, 0(t6)                    # ft4 = py[i]
    add t6, s6, s11
    flw ft5, 0(t6)                    # ft5 = pz[i]

    # Zero scalar accumulators for this i
    fmv.s ft6, ft2                    # ft6 = 0.0  
    fmv.s ft7, ft2                    # ft7 = 0.0  
    fmv.s ft8, ft2                    # ft8 = 0.0  

    mv t2, s4                         # t2 -> px[0]
    mv t3, s5                         # t3 -> py[0]
    mv t4, s6                         # t4 -> pz[0]
    mv t5, s10                        # t5 -> mass[0]
    li s3, 100                        # s3 = j_remaining
    li s2, 0                          # s2 = j_base index 

inner_j_loop:
    beqz s3, inner_j_done
    vsetvli t0, s3, e32, m1, ta, ma  
    vle32.v v1, (t2)                  # v1 = px[j..j+vl]
    vle32.v v2, (t3)                  # v2 = py[j..j+vl]
    vle32.v v3, (t4)                  # v3 = pz[j..j+vl]

    vfsub.vf v1, v1, ft3             # v1 = dx
    vfsub.vf v2, v2, ft4             # v2 = dy
    vfsub.vf v3, v3, ft5             # v3 = dz

    vfmul.vv v4, v1, v1              # v4 = dx^2
    vfmul.vv v5, v2, v2              # v5 = dy^2
    vfmul.vv v6, v3, v3              # v6 = dz^2
    vfadd.vv v4, v4, v5              # v4 = dx^2 + dy^2
    vfadd.vv v4, v4, v6              # v4 = r^2

    vfadd.vf v4, v4, ft1             # v4 = r^2 + soft^2

 
    vfsqrt.v  v5, v4                 
    vfmul.vv  v5, v5, v4             

    vle32.v   v6, (t5)               # v6 = mass
    vfmul.vf  v6, v6, ft0            # v6 = G * mass
    vfdiv.vv  v6, v6, v5             # v6 = acc_factor

    vid.v     v7                      
    vadd.vx   v7, v7, s2             
    vmseq.vx  v0, v7, s0             

    vfmerge.vfm v6, v6, ft2, v0     
    vfmul.vv  v7, v6, v1             # v7 = acc_factor * dx  
    vfmul.vv  v8, v6, v2             # v8 = acc_factor * dy
    vfmul.vv  v9, v6, v3             # v9 = acc_factor * dz


    vfmv.v.f  v11, ft2               
    vfredusum.vs v10, v7, v11        
    vfmv.f.s  ft9, v10              
    fadd.s    ft6, ft6, ft9          

    vfmv.v.f  v11, ft2
    vfredusum.vs v10, v8, v11        
    vfmv.f.s  ft9, v10
    fadd.s    ft7, ft7, ft9          # accumulate ay[i]

    vfmv.v.f  v11, ft2
    vfredusum.vs v10, v9, v11        # v10[0] = sum of az strip
    vfmv.f.s  ft9, v10
    fadd.s    ft8, ft8, ft9          # accumulate az[i]

    slli  t1, t0, 2                  
    add   t2, t2, t1                 
    add   t3, t3, t1                 
    add   t4, t4, t1                 
    add   t5, t5, t1                 
    add   s2, s2, t0                  
    sub   s3, s3, t0                 
    j inner_j_loop

inner_j_done:

    add t6, s7, s11
    fsw ft6, 0(t6)                   # ax[i] = ft6
    add t6, s8, s11
    fsw ft7, 0(t6)                   # ay[i] = ft7
    add t6, s9, s11
    fsw ft8, 0(t6)                   # az[i] = ft8

    addi s0, s0, 1                   # i++
    j outer_i_loop

acc_done:
    ld s11,   8(sp)
    ld s10,  16(sp)
    ld s9,   24(sp)
    ld s8,   32(sp)
    ld s7,   40(sp)
    ld s6,   48(sp)
    ld s5,   56(sp)
    ld s4,   64(sp)
    ld s3,   72(sp)
    ld s2,   80(sp)
    ld s1,   88(sp)
    ld s0,   96(sp)
    ld ra,  104(sp)
    addi sp, sp, 112
    ret
