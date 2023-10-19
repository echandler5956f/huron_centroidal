#!/home/quant/ros/ocs2_ws/src/huron_centroidal/.venv/bin/python3
import rospy
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import JointState
from std_srvs.srv import Empty, EmptyRequest

import time
from types import SimpleNamespace

import casadi
# import example_robot_data as robex
import numpy as np
import pinocchio as pin
from pinocchio import casadi as cpin
from pinocchio.robot_wrapper import RobotWrapper

from utils.meshcat_viewer_wrapper import MeshcatVisualizer

xc = None
vc = None
tauc = None

def joint_callback(data):
    global xc, vc, tauc
    xc = data.position
    vc = data.velocity

def main():
    global xc, vc, tauc

    rospy.wait_for_service('/gazebo/unpause_physics')
    unpause_physics_client = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)

    effortController = rospy.Publisher('/huron/joint_group_effort_controller/command', Float64MultiArray, queue_size=100)
    rospy.Subscriber("/huron/joint_states", JointState, joint_callback, queue_size=100)

    rospy.init_node('huron_centroidal', anonymous=True)

    # Change numerical print
    pin.SE3.__repr__ = pin.SE3.__str__
    np.set_printoptions(precision=2, linewidth=300, suppress=True, threshold=1e6)

    ### HYPER PARAMETERS
    Mtarget = pin.SE3(pin.utils.rotate("y", np.pi/2), np.array([0.15, -0.15, 0.2]))  # x,y,z
    contacts = [SimpleNamespace(name="l_foot_v_ft_link", type=pin.ContactType.CONTACT_6D)]
    fixedFootFrameName = "l_foot_v_ft_link"
    endEffectorFrameName = "r_foot_v_ft_link"

    # --- Load robot model
    builder = RobotWrapper.BuildFromURDF
    robot = builder(
                    "/home/quant/ros/ocs2_ws/src/HURON-Model/huron_description/urdf/huron_cheat.urdf",
                    ["/home/quant/ros/ocs2_ws/src/HURON-Model/huron_description"],
                    None,
                )
    robot.q0 = np.array([0, 0, 1.0627, 0, 0, 0, 1,
                        0.0000, 0.0152, -0.2884, 0.7541, -0.4657, -0.0152,
                        0.0000, -0.0152, -0.2884, 0.7541, -0.4657, 0.0152])

    # Open the viewer
    viz = MeshcatVisualizer(robot)
    viz.display(robot.q0)

    # The pinocchio model is what we are really interested by.
    model = robot.model
    data = model.createData()

    pin.computeTotalMass(model, data)
    pin.framesForwardKinematics(model, data, robot.q0)
    # Hyperparameters for the control
    Kp = 5.             # proportional gain (P of PD)
    Kv = 2 * np.sqrt(Kp) # derivative gain (D of PD)
    T = 100
    DT = 0.002
    mu = 0.7
    g = np.array([0, 0, 9.81])
    m = data.mass[0]
    fg = m * g

    fixedFoot_ID = model.getFrameId(fixedFootFrameName)
    endEffector_ID = model.getFrameId(endEffectorFrameName)
    fixed_foot = data.oMf[fixedFoot_ID]
    for c in contacts:
        c.id = model.getFrameId(c.name)
        assert c.id < len(model.frames)
        c.jid = model.frames[c.id].parentJoint
        c.placement = model.frames[c.id].placement
        c.model = pin.RigidConstraintModel(c.type, model, c.jid, c.placement)
    # contact_models = [c.model for c in contacts]

    # contact_datas = [c.createData() for c in contact_models]

    # pin.initConstraintDynamics(model, data, contact_models)
    # q = robot.q0.copy()
    # v = np.zeros(model.nv)
    # tau = np.zeros(model.nv)
    # pin.constraintDynamics(model, data, q, v, tau, contact_models, contact_datas)

    # --- Add box to represent target
    # Add a vizualization for the target
    boxID = "world/box"
    viz.addBox(boxID, [0.05, 0.1, 0.2], [1.0, 0.2, 0.2, 0.5])
    # Add a vizualisation for the tip of the arm.
    tipID = "world/blue"
    viz.addBox(tipID, [0.08] * 3, [0.2, 0.2, 1.0, 0.5])
    for c in contacts:
        c.viz = f"world/contact_{c.name}"
        viz.addSphere(c.viz, [0.07], [0.8, 0.8, 0.2, 0.5])


    def displayScene(q, dt=1e-1):
        """
        Given the robot configuration, display:
        - the robot
        - a box representing endEffector_ID
        - a box representing Mtarget
        """
        pin.framesForwardKinematics(model, data, q)
        M = data.oMf[endEffector_ID]
        viz.applyConfiguration(boxID, Mtarget)
        viz.applyConfiguration(tipID, M)
        for c in contacts:
            viz.applyConfiguration(c.viz, data.oMf[c.id])
        viz.display(q)
        time.sleep(dt)


    def displayTraj(qs, dt=1e-1):
        for q in qs[1:]:
            displayScene(q, dt=dt)


    displayScene(robot.q0)

    # --- Casadi helpers
    cmodel = cpin.Model(model)
    cdata = cmodel.createData()

    nq = model.nq # 19
    nv = model.nv # 18
    nm = 3
    nh = nm + nm
    nx = 2 * nh + nq + nv # 49
    ndx = 2 * nh + 2 * nv # 48
    cx = casadi.SX.sym("x", nx, 1)
    ch = cx[:nh] # momenta
    clin = ch[:nm] # first 3 are linear momentum
    cang = ch[nm:] # last 3 are angular momentum
    cq = cx[nh:nq+nh] # next 19 of x are q
    cqb = cq[:7] # first 7 of q are qb
    cqj = cq[7:] # last 12 of q are qj
    cdh = cx[nq+nh:nq+nh+nh] # next 6 are derivative of momenta
    cdlin = cdh[:nm] # first 3 are derivative linear momentum
    cdang = cdh[nm:] # last 3 are derivative of angular momentum
    cv = cx[nq+nh+nh:] # last 18 of x are v
    cvb = cv[:6] # first 6 of v are vb
    cvj = cv[6:] # last 12 of v are vj

    cdx = casadi.SX.sym("dx", ndx, 1)
    cdh_ = cdx[:nh]
    cdq = cdx[nh:nv+nh]
    cddh_ = cdx[nv+nh:nv+nh+nh]
    cdv = cdx[nv+nh+nh:]

    cu = casadi.SX.sym("u", 6, 1)
    cf = cu[:3] # first 3 are contact forces
    ctau = cu[3:] # last 3 are contact torques

    cvj_ = casadi.SX.sym("vjs", nv - 6, 1)

    # print(ch)
    # print(clin)
    # print(cang)
    # print(cq)
    # print(cqb)
    # print(cqj)
    # print(cdh)
    # print(cdlin)
    # print(cdang)
    # print(cv)
    # print(cvb)
    # print(cvj)

    # Compute casadi graphs
    cpin.centerOfMass(cmodel, cdata, cq, False)
    cpin.computeCentroidalMap(cmodel, cdata, cq)
    cpin.forwardKinematics(cmodel, cdata, cq, cv)
    cpin.updateFramePlacements(cmodel, cdata)

    # Sym graph for the integration operation x,dx -> x(+)dx = [model.integrate(q,dq),v+dv]
    cintegrate = casadi.Function(
        "integrate",
        [cx, cdx],
        [casadi.vertcat(ch + cdh_,
                        cpin.integrate(cmodel, cq, cdq), 
                        cdh + cddh_,
                        cv + cdv)],
    )

    Ag = cdata.Ag
    # print(Ag.shape)
    Agb = Ag[:,:6]
    Agj = Ag[:,6:]
    # print(Agb.shape)
    # print(Agj.shape)
    # test = Agj*cvj
    # print(test.shape)
    # Sym graph for the integration operation x' = [ q+vDT+aDT**2, v+aDT ]
    cnext = casadi.Function(
        "next",
        [cx, cu, cvj_],
        [
            casadi.vertcat(
                clin + cdlin * DT,
                cang + cdang * DT,
                cpin.integrate(cmodel, cq, cv * DT),
                (cf - fg)/m,
                (casadi.cross(cdata.oMf[fixedFoot_ID].translation - cdata.com[0], cf) + ctau)/m,
                casadi.mtimes(casadi.inv(Agb), (m * ch - casadi.mtimes(Agj, cvj_))),
                cvj_,
            )
        ],
    )

    # Sym graph for the operational error
    error_tool = casadi.Function(
        "etool3", [cx], [cdata.oMf[endEffector_ID].translation - Mtarget.translation]
    )

    error_fixed_foot = casadi.Function(
        "efoot3", [cx], [cdata.oMf[fixedFoot_ID].translation - fixed_foot.translation]
    )

    ### PROBLEM

    opti = casadi.Opti()
    var_dxs = [opti.variable(ndx) for t in range(T + 1)]
    var_us = [opti.variable(6) for t in range(T)]
    var_vjs = [opti.variable(nv-6) for t in range(T)]
    var_xs = [
        cintegrate(np.concatenate([np.zeros(nh), robot.q0, np.zeros(nh), np.zeros(nv)]), var_dx) for var_dx in var_dxs
    ]

    totalcost = 0
    # Define the running cost
    for t in range(T):
        totalcost += 1e-3 * DT * casadi.sumsqr(var_xs[t][nh+nq+nh:]) # penalize velocities
        totalcost += 1e-6 * DT * casadi.sumsqr(var_us[t][:3]) # penalize contact forces
        # totalcost += 1e-3 * DT * casadi.sumsqr(var_us[t][3:]) # penalize contact torques
    totalcost += 1e1 * casadi.sumsqr(error_tool(var_xs[T]))

    opti.subject_to(var_xs[0][nh:nq+nh] == robot.q0)
    opti.subject_to(var_xs[0][nh+nq+nh:] == 0)  # zero initial velocity
    opti.subject_to(var_xs[T][nh+nq+nh:] == 0)  # zero terminal velocity

    # Define the integration constraints
    for t in range(T):
        opti.subject_to(error_fixed_foot(var_xs[t]) == 0)
        opti.subject_to(cnext(var_xs[t], var_us[t], var_vjs[t]) == var_xs[t + 1])
        opti.subject_to(var_us[t][0]/var_us[t][2] >= -mu)
        opti.subject_to(var_us[t][0]/var_us[t][2] <= mu)
        opti.subject_to(var_us[t][1]/var_us[t][2] >= -mu)
        opti.subject_to(var_us[t][1]/var_us[t][2] <= mu)
        opti.subject_to(var_us[t][2] >= 0)
        opti.subject_to(var_us[t][3] == 0)
        opti.subject_to(var_us[t][4] == 0)
        opti.set_initial(var_us[t][2], fg[2])

    # time.sleep(4)

    ### SOLVE
    opti.minimize(totalcost)
    p_opts = {"expand": True}
    s_opts = {"max_iter": 100}
    opti.solver("ipopt", p_opts, s_opts) # set numerical backend
    opti.callback(lambda i: displayScene(opti.debug.value(var_xs[-1][nh:nq+nh])))

    # Caution: in case the solver does not converge, we are picking the candidate values
    # at the last iteration in opti.debug, and they are NO guarantee of what they mean.
    try:
        sol = opti.solve_limited()
        sol_xs = [opti.value(var_x) for var_x in var_xs]
    except:
        print("ERROR in convergence, plotting debug info.")
        sol_xs = [opti.debug.value(var_x) for var_x in var_xs]

    print("***** Display the resulting trajectory ...")
    
    # xsol = [x[nh:nq+nh] for x in sol_xs]
    # vsol = [x[nh+nq:nh+nq+nv] for x in sol_xs]
    # while True:
    #     # displayScene(robot.q0, 1)
    #     displayTraj(xsol, DT)
    #     xsol.reverse()
    #     displayTraj(xsol, DT)
    #     xsol.reverse()

    xsol = [x[nh+7:nq+nh] for x in sol_xs]
    vsol = [x[nh+6+nq:nh+nq+nv] for x in sol_xs]
    asol = [2 * ((xsol[t+1] - xsol[t])/(DT*DT) - (vsol[t]/DT)) for t in range(T-1)]

    unpause_physics_client(EmptyRequest())
    rospy.wait_for_message("/huron/joint_states", JointState, timeout=None)
    for t in range(T-1):
        tstart = rospy.Time.now().to_sec()
        tauc = asol[t] + Kp * (xsol[t] - xc) + Kv * (vsol[t] - vc) 
        while rospy.Time.now().to_sec() < tstart + DT:
            effortController.publish(Float64MultiArray(data=tauc))
        # xsol.reverse()
        # vsol.reverse()
        # asol.reverse()
        # for t in range(T-1):
        #     tauc = asol[t] + Kp * (xsol[t] - xc) + Kv * (vsol[t] - vc) 
        #     effortController.publish(Float64MultiArray(data=tauc))
        #     time.sleep(DT)
        # xsol.reverse()
        # vsol.reverse()
        # asol.reverse()


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
    # main()