#!/home/quant/ros_ws/src/huron_centroidal/.venv/bin/python3

import casadi
import numpy as np
import pinocchio as pin
from dataclasses import dataclass
from pinocchio import casadi as cpin


@dataclass
class Phase():
    frame_id: int
    type_6D: bool
    in_contact: bool
    

class ContactPhase():
    def __init__(s, phases, steps, period):
        s.phases = phases
        s.id_list = [phase.frame_id for phase in phases]
        s.steps = steps
        s.period = period
        s.contact_list = [int(phase.in_contact) for phase in phases]


class ContactSequence():
    def __init__(s, contacts):
        s.contacts = contacts
        s.contact_phase_list = [contact.contact_list for contact in s.contacts]
        s.num_cons = len(s.contacts)
        s.num_ee = len(s.contact_phase_list[0])
        s.num_u_list = [s.LegTypeToSize(phase.type_6D)for phase in s.contacts[0].phases]
        s.num_u = np.sum(s.num_u_list)
        
        s.t_list = [contact.period for contact in s.contacts]
        s.Tt = np.sum(s.t_list)
        s.cum_tt = np.cumsum(s.t_list)
        
        s.step_list = [contact.steps for contact in s.contacts]
        s.Ts = np.sum(s.step_list)
        s.cum_ts = np.cumsum(s.step_list)
        
    def InContact(s, ee, k):
        s.contact_phase_list[s.GetCurrentPhase(k)][ee]
    
    def GetFrameIDList(s):
        return s.contacts[0].id_list

    def GetLegType(s, ee):
        return s.contacts[0].phases[ee].type_6D
    
    def LegTypeToSize(s, legtype):
        return (int(legtype) + 1) * 3
    
    def GetLegSize(s, ee):
        return s.LegTypeToSize(s.GetLegType(ee))
        
    def GetOffset(s, ee):
        return ee * s.Ts
        
    def GetCurrentPhase(s, k):
        i = 0
        for j in range(s.num_cons):
            i = i + (k >= s.cum_ts[j])
        return i
    
    def IsNewContact(s, k, eeindex):
        i = s.GetCurrentPhase(k)
        if s.contact_phase_list[i][eeindex] and i > 0:
            if not s.contact_phase_list[i - 1, eeindex] and k - s.cum_ts[i - 1] == 0:
                return True
        return False
    

class PseudoSpectralCollocation():
    def __init__(s, degree):
        # Degree of interpolating polynomial
        s.degree = degree

        # Get collocation points
        tau_root = np.append(0, casadi.collocation_points(s.degree, 'legendre'))

        # Coefficients of the collocation equation
        s.C = np.zeros((s.degree + 1, s.degree + 1))

        # Coefficients of the continuity equation
        s.D = np.zeros(s.degree + 1)

        # Coefficients of the quadrature function
        s.B = np.zeros(s.degree + 1)

        # Construct polynomial basis
        for j in range(s.degree + 1):
            # Construct Lagrange polynomials to get the polynomial basis at the collocation point
            p = np.poly1d([1])
            for r in range(s.degree + 1):
                if r != j:
                    p *= np.poly1d([1, -tau_root[r]]) / (tau_root[j] - tau_root[r])

            # Evaluate the polynomial at the final time to get the coefficients of the continuity equation
            s.D[j] = p(1.0)

            # Evaluate the time derivative of the polynomial at all collocation points to get the coefficients of the
            # continuity equation
            p_der = np.polyder(p)
            for r in range(s.degree + 1):
                s.C[j, r] = p_der(tau_root[r])

            # Evaluate the integral of the polynomial to get the coefficients of the quadrature function
            pint = np.polyint(p)
            s.B[j] = pint(1.0)


class Parameters():
    def __init__(s):
        s.g = np.array([0, 0, 9.81])
        s.mu = 0.7
        s.p_opts = {"expand": True}
        s.s_opts = {"max_iter": 50}
        

class CentroidalTrajOpt():
    
    def __init__(s, model, params, contact_sequence, collocation, q0):
        
        s.p = params
        s.cons = contact_sequence
        s.col = collocation
        s.q0 = q0
        s.model = model
        s.data = s.model.createData()
        pin.computeTotalMass(s.model, s.data)

        s.m = s.data.mass[0]
        s.N = s.cons.Ts
        
        s.cmodel = cpin.Model(s.model)
        s.cdata = s.cmodel.createData()
        
        s.nq = s.model.nq
        s.nv = s.model.nv
        nm = 3
        s.nh = nm + nm
        s.nx = 2 * s.nh + s.nq + s.nv
        s.ndx = 2 * s.nh + 2 * s.nv
        
        s.cx = casadi.SX.sym("x", s.nx, 1)
        s.cdx = casadi.SX.sym("dx", s.ndx, 1)
        
        s.cu = casadi.SX.sym("u", s.contact_sequence.num_u, 1)
        s.cgu = casadi.SX.sym("gu", 6, 1)
        s.cvj = casadi.SX.sym("vj", s.nv - 6, 1)
        s.DT = casadi.SX.sym("dt", 1, 1)

        s.ch = s.cx[:s.nh] # momenta: nh x 1
        s.cdh = s.cx[s.nq+s.nh:s.nq+s.nh+s.nh] # momenta time derivative: nh x 1
        s.cdh_ = s.cdx[:s.nh] # momenta delta: nh x 1
        s.cddh_ = s.cdx[s.nv+s.nh:s.nv+s.nh+s.nh] # momenta time derivative delta: nh x 1
        s.cq = s.cx[s.nh:s.nq+s.nh] # q: nq x 1
        s.cdq = s.cdx[s.nh:s.nv+s.nh] # q delta: nv x 1
        s.cv = s.cx[s.nq+s.nh+s.nh:] # v: nv x 1
        s.cdv = s.cdx[s.nv+s.nh+s.nh:] # v delta: nv x 1
        s.cf = s.cgu[:3] # f: 3 x 1
        s.ctau = s.cgu[3:] # tau: 3 x 1
        
        s.T = s.cons.t_list # replace with sym
        
        s.xdot = None
        s.L = None
    
    def Get_dt_k(s, k):
        i = s.cons.GetCurrentPhase(k)
        return i, s.T[i] / s.cons.step_list[i]
    
    def ComputeCasadiGraphs(s):
        cpin.centerOfMass(s.cmodel, s.cdata, s.cq, False)
        cpin.computeCentroidalMap(s.cmodel, s.cdata, s.cq)
        cpin.forwardKinematics(s.cmodel, s.cdata, s.cq, s.cv)
        cpin.updateFramePlacements(s.cmodel, s.cdata)
        Ag = s.cdata.Ag
        s.cintegrate = casadi.Function(
            "integrate",
            [s.cx, s.cdx],
            [casadi.vertcat(s.ch + s.cdh_,
                            cpin.integrate(s.cmodel, s.cq, s.cdq), 
                            s.cdh + s.cddh_,
                            s.cv + s.cdv)],
        )
        s.xdot = casadi.Function(
            "xdot",
            [s.cx, s.cgu, s.cvj, s.DT],
            [
                casadi.vertcat(
                    s.cdh,
                    cpin.integrate(s.cmodel, s.cq, s.cv),
                    (s.cf - s.m*s.p.g)/s.m,
                    (s.ctau)/s.m,
                    casadi.mtimes(casadi.inv(Ag[:,:6]), (s.m * s.ch - casadi.mtimes(Ag[:,6:], s.cvj))),
                    s.cvj,
                )
            ],
        )
        
        s.L = casadi.Function(
            "L",
            [s.cx, s.cgu, s.cvj],
            [
                casadi.sumsqr(s.cv) + casadi.sumsqr(s.cf) + casadi.sumsqr(s.ctau)
            ],
        )

        # Sym graph for the contact constraint and Baugart correction terms
        # Works for both 3D and 6D contacts.
        # Uses the contact list <contacts> where each item must have a <name>, an <id> and a <type> field.
        s.dpcontacts = {}  # Error in contact position
        s.vcontacts = {}  # Error in contact velocity

        for ee in range(s.cons.num_ee):
            cid = s.cons.GetFrameIDList()[ee]
            name = str(cid)
            if not s.cons.GetLegType():
                p0 = s.data.oMf[cid].translation.copy()
                s.dpcontacts[ee] = casadi.Function(
                    f"dpcontact_{name}",
                    [s.cx],
                    [-(s.cdata.oMf[cid].inverse().act(casadi.SX(p0)))],
                )
                s.vcontacts[ee] = casadi.Function(
                    f"vcontact_{name}",
                    [s.cx],
                    [cpin.getFrameVelocity(s.cmodel, s.cdata, cid, pin.LOCAL).linear],
                )
            elif s.cons.GetLegType():
                p0 = s.data.oMf[cid]
                s.dpcontacts[ee] = casadi.Function(f"dpcontact_{name}", [s.cx], [np.zeros(6)])
                s.vcontacts[ee] = casadi.Function(
                    f"vcontact_{name}",
                    [s.cx],
                    [cpin.getFrameVelocity(s.cmodel, s.cdata, cid, pin.LOCAL).vector],
                )
        
    
    def SetupProblem(s):
        opti = casadi.Opti()
        var_dxs = [opti.variable(s.ndx) for k in range(s.N + 1)]
        s.var_us = []
        s.u_idxs = []
        var_vjs = [opti.variable(s.nv-6) for k in range(s.N)]
        var_xs = [s.cintegrate(np.concatenate([np.zeros(s.nh), s.q0, np.zeros(s.nh), np.zeros(s.nv)]), var_dx) for var_dx in var_dxs]
        
        totalcost = 0
        for ee in range(s.cons.num_ee):
            s.u_idx.append([])
            for k in range(s.N):
                offset = s.cons.GetOffset(ee)
                legsize = s.cons.GetLegSize(ee)
                if s.cons.InContact(ee, k):
                    s.var_us.append(opti.variable(legsize))
                    s.u_idxs[ee].extend(offset + k)
                else:
                    s.var_us.append(casadi.SX.zeros(legsize))
        
        x_k = var_xs[0]
        for k in range(s.N):
            i, dt = s.Get_dt_k(k)
            
            grf = np.zeros((3, 1))
            tau = np.zeros((3, 1))
            for ee in range(s.cons.num_ee):
                offset = s.cons.GetOffset(ee)
                F = s.var_us[offset + k]
                if s.cons.InContact(ee, k):
                    grf += F[:3]
                    tau += casadi.cross(s.cdata.oMf[s.cons.GetFrameIDList()[ee]].translation.vector, F[:3])
                    # add contact constraints
                    opti.subject_to(s.vcontacts[ee](x_k) == 0)
                    opti.subject_to(F[0]/F[2] >= -s.p.mu)
                    opti.subject_to(F[0]/F[2] <= s.p.mu)
                    opti.subject_to(F[1]/F[2] >= -s.p.mu)
                    opti.subject_to(F[1]/F[2] <= s.p.mu)
                    opti.subject_to(F[2] >= 0)
                    opti.set_initial(F[2], s.m * s.p.g[2])
                    if s.cons.GetLegSize(ee) == 6:
                        tau = tau + F[3:]
                else:
                    pass
                    # add swing constraints
                    # opti.subject_to(vel_foot(var_xs[t]) == 0)
            u_k = casadi.vertcat(grf, tau)
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
                opti.subject_to(dt*fj - x_p == 0)
                # Add contribution to the end state
                x_k_end += s.col.D[j] * x_c[j-1]
                # Add contribution to quadrature function
                totalcost += s.col.B[j] * qj * dt
                
            if k < s.N:
                # New NLP variable for state at end of interval
                x_k = var_xs[k + 1]
                # Add equality constraint
                opti.subject_to(x_k_end - x_k == 0)
            
            
        ### SOLVE
        opti.minimize(totalcost)
        opti.solver("ipopt", s.p_opts, s.s_opts) # set numerical backend

        # Caution: in case the solver does not converge, we are picking the candidate values
        # at the last iteration in opti.debug, and they are NO guarantee of what they mean.
        try:
            sol = opti.solve_limited()
            sol_xs = [opti.value(var_x) for var_x in var_xs]
        except:
            print("ERROR in convergence, plotting debug info.")
            sol_xs = [opti.debug.value(var_x) for var_x in var_xs]            
            