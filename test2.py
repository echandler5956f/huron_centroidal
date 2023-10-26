# import casadi as ca
# import numpy as np
# import pinocchio as pin
# import matplotlib.pyplot as plt
# from pinocchio import casadi as cpin

# def quaternion_product(q1, q2):
#     w1, x1, y1, z1 = q1[0], q1[1], q1[2], q1[3]
#     w2, x2, y2, z2 = q2[0], q2[1], q2[2], q2[3]
    
#     w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
#     x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
#     y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
#     z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    
#     return ca.vertcat(w, x, y, z)

# def exp_map(omega):
#     omega_norm = ca.norm_2(omega)
#     q_w = ca.cos(omega_norm/2)
#     q_xyz = ca.if_else(omega_norm > 1e-10, ca.sin(omega_norm/2) * omega / omega_norm, omega / 2)
#     return ca.vertcat(q_w, q_xyz)

# def log_map(q):
#     q_xyz = q[1:4]
#     omega_norm = 2 * ca.acos(q[0])
#     return ca.if_else(omega_norm > 1e-10, omega_norm * q_xyz / ca.norm_2(q_xyz), 2*q_xyz)

# d = 3
# tau_root = np.append(0, ca.collocation_points(d, 'legendre'))
# C = np.zeros((d+1,d+1))
# D = np.zeros(d+1)
# B = np.zeros(d+1)

# for j in range(d+1):
#     p = np.poly1d([1])
#     for r in range(d+1):
#         if r != j:
#             p *= np.poly1d([1, -tau_root[r]]) / (tau_root[j]-tau_root[r])
#     D[j] = p(1.0)
#     pder = np.polyder(p)
#     for r in range(d+1):
#         C[j,r] = pder(tau_root[r])
#     pint = np.polyint(p)
#     B[j] = pint(1.0)

# T = 10.
# tangent_vec = ca.MX.sym('tangent_vec', 3)  # 3D tangent vector 
# u = ca.MX.sym('u', 3)

# quaternion = exp_map(tangent_vec)
# half_q = 0.5 * quaternion
# omega = ca.vertcat(0, u[0], u[1], u[2])
# xdot = log_map(quaternion_product(half_q, omega))
# L = ca.mtimes(u.T, u)

# f = ca.Function('f', [tangent_vec, u], [xdot, L], ['tangent_vec', 'u'], ['xdot', 'L'])

# N = 20 
# h = T/N
# w = []
# w0 = []
# lbw = []
# ubw = []
# J = 0
# g = []
# lbg = []
# ubg = []

# tangent_0 = ca.MX.sym('Tangent0', 3)
# w.append(tangent_0)
# lbw += [[0, 0, 0]]  
# ubw += [[0, 0, 0]]
# w0 += [[0, 0, 0]]

# for k in range(N):
#     Uk = ca.MX.sym('U_' + str(k), 3)
#     w.append(Uk)
#     lbw += [[-10, -10, -10]]
#     ubw += [[10, 10, 10]]
#     w0 += [[0, 0, 0]]

#     Xc = []
#     for j in range(d):
#         T_kj = ca.MX.sym('T_'+str(k)+'_'+str(j), 3)
#         Xc.append(T_kj)
#         w.append(T_kj)
#         lbw += [[-np.pi, -np.pi, -np.pi]]
#         ubw += [[np.pi, np.pi, np.pi]]
#         w0 += [[0, 0, 0]]

#     Xk_end = D[0]*tangent_0
#     for j in range(1, d+1):
#         xp = C[0,j]*tangent_0
#         for r in range(d): xp = xp + C[r+1,j]*Xc[r]
#         fj, qj = f(Xc[j-1], Uk)
#         g.append(h*fj - xp)
#         lbg += [[0, 0, 0]]
#         ubg += [[0, 0, 0]]
#         Xk_end = Xk_end + D[j]*Xc[j-1]
#         J = J + B[j]*qj*h

#     tangent_next = ca.MX.sym('T_' + str(k+1), 3)
#     w.append(tangent_next)
#     lbw += [[-np.pi, -np.pi, -np.pi]]
#     ubw += [[np.pi, np.pi, np.pi]]
#     w0 += [[0, 0, 0]]
#     g.append(Xk_end - tangent_next)
#     lbg += [[0, 0, 0]]
#     ubg += [[0, 0, 0]]

# w = ca.vertcat(*w)
# g = ca.vertcat(*g)

# w0 = np.concatenate([np.array(item).reshape(-1) for item in w0])
# lbw = np.concatenate([np.array(item).reshape(-1) for item in lbw])
# ubw = np.concatenate([np.array(item).reshape(-1) for item in ubw])
# lbg = np.concatenate([np.array(item).reshape(-1) for item in lbg])
# ubg = np.concatenate([np.array(item).reshape(-1) for item in ubg])

# prob = {'f': J, 'x': w, 'g': g}
# solver = ca.nlpsol('solver', 'ipopt', prob)

# # trajectories = ca.Function('trajectories', [w], [x_plot, u_plot], ['w'], ['x', 'u'])

# sol = solver(x0=w0, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)



import numpy as np
import casadi as ca
import pinocchio as pin
from dataclasses import dataclass
from pinocchio import casadi as cpin



class CentroidalTrajOpt:
    def __init__(s, model, data, viz, params, contact_sequence, collocation):
        s.model = model
        s.data = data
        s.viz = viz
        s.p = params
        s.cons = contact_sequence
        s.col = collocation
        s.q0 = s.p.q0
        pin.computeTotalMass(s.model, s.data)

        s.Mtarget = pin.SE3(
            pin.utils.rotate("y", np.pi / 2), np.array([0.0775, 0.05, 0.1])
        )  # x,y,z

        s.m = s.data.mass[0]
        s.N = s.cons.cumulative_knots[-1]

        s.cmodel = cpin.Model(s.model)
        s.cdata = s.cmodel.createData()

        s.nq = s.model.nq
        s.nv = s.model.nv
        nm = 3
        s.nh = nm + nm
        s.nx = s.nh + s.nq
        s.ndx = s.nh + s.nv

        s.cx = ca.SX.sym("x", s.nx, 1)
        s.cdx = ca.SX.sym("dx", s.ndx, 1)

        s.cgu = ca.SX.sym("gu", 6, 1)
        s.cvj = ca.SX.sym("vj", s.nv - 6, 1)

        s.ch = s.cx[: s.nh]  # momenta: nh x 1
        s.cdh = s.cdx[: s.nh]  # momenta delta: nh x 1
        s.cq = s.cx[s.nh : s.nq + s.nh]  # q: nq x 1
        s.cdq = s.cdx[s.nh : s.nv + s.nh]  # q delta: nv x 1
        s.cdqj = s.cdq[6:]  # q_j delta: (nv - 6) x 1
        s.cf = s.cgu[:3]  # f: 3 x 1
        s.ctau = s.cgu[3:]  # tau: 3 x 1

        s.cintegrate = None
        s.xdot = None
        s.L = None
        s.M = None
        s.dpcontacts = None

    def compute_casadi_graphs(s):
        cpin.centerOfMass(s.cmodel, s.cdata, s.cq, False)
        cpin.computeCentroidalMap(s.cmodel, s.cdata, s.cq)
        cpin.forwardKinematics(s.cmodel, s.cdata, s.cq)
        cpin.updateFramePlacements(s.cmodel, s.cdata)
        Ag = s.cdata.Ag

        # Integrate over the Lie Group
        s.cintegrate = ca.Function(
            "integrate",
            [s.cx, s.cdx],
            [ca.vertcat(s.ch + s.cdh, cpin.integrate(s.cmodel, s.cq, s.cdq))],
        )

        # Centroidal dynamics in continuous time
        s.xdot = ca.Function(
            "xdot",
            [s.cx, s.cgu, s.cvj],
            [
                ca.vertcat(
                    (s.cf - s.m * s.p.g) / s.m,
                    (s.ctau) / s.m,
                    ca.mtimes(
                        ca.inv(Ag[:, :6]), (s.m * s.ch - ca.mtimes(Ag[:, 6:], s.cvj))
                    ),
                    s.cvj,
                )
            ],
        )

        s.L = ca.Function(
            "L",
            [s.cx, s.cgu, s.cvj],
            [1e-3 * ca.sumsqr(s.cvj) + 1e4 * ca.sumsqr(s.cf)],
        )

        s.M = ca.Function(
            "M",
            [s.cx],
            [
                1e4
                * ca.sumsqr(
                    s.Mtarget.translation
                    - s.cdata.oMf[s.model.getFrameId("r_foot_v_ft_link")].translation
                )
            ],
        )

        s.dpcontacts = {}  # Error in contact position

        for ee in s.cons.get_all_end_effectors():
            cid = ee.frame_id
            name = str(cid)
            p0 = s.data.oMf[cid].translation.copy()
            s.dpcontacts[ee] = ca.Function(
                f"dpcontact_{name}",
                [s.cx],
                [-(s.cdata.oMf[cid].inverse().act(ca.SX(p0)))],
            )

    def setup_problem(s):
        opti = ca.Opti()
        var_dxs = [opti.variable(s.ndx) for k in range(s.N + 1)]
        var_dts = []
        dts = {}
        var_us = []
        us = defaultdict(list)
        var_vjs = [opti.variable(s.nv - 6) for k in range(s.N)]
        var_xs = [
            s.cintegrate(np.concatenate([np.zeros(s.nh), s.q0]), var_dx)
            for var_dx in var_dxs
        ]

        totalcost = 0
        for ee in s.cons.get_all_end_effectors():
            us[ee] = []
            for k in range(s.N):
                legsize = s.cons.get_contact_size(ee)
                if s.cons.is_in_contact(ee, k):
                    u = opti.variable(legsize)
                    var_us.append(u)
                    us[ee].append(u)
                else:
                    us[ee].append(np.zeros(legsize))

        for phase in s.cons.sequence:
            if phase.timing_var:
                dt_var = opti.variable(1)
                var_dts.append(dt_var)
                dts[phase] = dt_var
            else:
                dts[phase] = phase.fixed_timing

        x_k = var_xs[0]
        for k in range(s.N):
            phase = s.cons.get_phase(k)
            dt = dts[phase]

            grf = np.zeros(3)
            tau = np.zeros(3)
            for ee in s.cons.get_all_end_effectors():
                F = us[ee][k]
                if s.cons.is_in_contact(ee, k):
                    grf += F[:3]
                    # tau += ca.cross(
                    #     s.cdata.oMf[s.cons.get_frame_id(ee)].translation,
                    #     F[:3],
                    # )
                    # add contact constraints
                    opti.subject_to(s.dpcontacts[ee](x_k) == 0)
                    opti.subject_to(F[0] / F[2] >= -s.p.mu)
                    opti.subject_to(F[0] / F[2] <= s.p.mu)
                    opti.subject_to(F[1] / F[2] >= -s.p.mu)
                    opti.subject_to(F[1] / F[2] <= s.p.mu)
                    opti.subject_to(F[2] >= 0)
                    opti.set_initial(F[2], s.m * s.p.g[2])
                    if s.cons.get_contact_size(ee) == 6:
                        tau = tau + F[3:]
                else:
                    pass
                    # add swing constraints
            u_k = ca.vertcat(grf, tau)
            vj_k = var_vjs[k]

            # State at collocation points
            x_c = []
            for j in range(s.col.degree):
                x_kj = opti.variable(s.nx)
                x_c.append(x_kj)

            # Loop over collocation points
            x_k_end = s.col.D[0] * x_k
            for j in range(s.col.degree + 1):
                # Expression for the state derivative at the collocation point
                x_p = s.col.C[0, j] * x_k
                for r in range(s.col.degree):
                    x_p += s.col.C[r + 1, j] * x_c[r]

                # Append collocation equations
                fj = s.xdot(x_c[j - 1], u_k, vj_k)
                qj = s.L(x_c[j - 1], u_k, vj_k)
                # By multiplying fj with h, we're effectively rescaling the
                # state's rate of change from normalized time back to the actual
                # time scale of the problem
                opti.subject_to(dt * fj - x_p == 0)
                # Add contribution to the end state
                x_k_end += s.col.D[j] * x_c[j - 1]
                # Add contribution to quadrature function
                totalcost += s.col.B[j] * qj * dt

            if k < s.N:
                # New NLP variable for state at end of interval
                x_k = var_xs[k + 1]
                # Add equality constraint
                opti.subject_to(x_k_end - x_k == 0)

        totalcost += s.M(x_k)

        # SOLVE
        opti.minimize(totalcost)
        opti.solver("ipopt", s.p.p_opts, s.p.s_opts)  # set numerical backend
