#!/home/quant/ros/ocs2_ws/src/huron_centroidal/.venv/bin/python3
import rospy
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import JointState
from std_srvs.srv import Empty, EmptyRequest
from gazebo_msgs.srv import GetModelState, GetModelStateRequest

import time
from types import SimpleNamespace

import casadi
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
    rospy.wait_for_service('/gazebo/pause_physics')
    rospy.wait_for_service('gazebo/get_model_state')
    unpause_physics_client = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
    pause_physics_client = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
    get_model_state_client = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)

    effortController = rospy.Publisher('/huron/joint_group_effort_controller/command', Float64MultiArray, queue_size=100)
    rospy.Subscriber("/huron/joint_states", JointState, joint_callback, queue_size=100)

    rospy.init_node('huron_centroidal', anonymous=True)

    # Change numerical print
    pin.SE3.__repr__ = pin.SE3.__str__
    np.set_printoptions(precision=5, linewidth=300, suppress=True, threshold=1e6)

    ### HYPER PARAMETERS
    Mtarget = pin.SE3(pin.utils.rotate("y", np.pi/2), np.array([0.0775, 0.05, 0.1]))  # x,y,z
    contacts = [SimpleNamespace(name="l_foot_v_ft_link", type=pin.ContactType.CONTACT_6D)]
    baseFrameName = "base"
    endEffectorFrameName = "r_foot_v_ft_link"
    fixedFootFrameName = "l_foot_v_ft_link"
    T = 50
    DT = 0.002
    Kp = 1e-3
    Kv = 1e-3
    Kpj = 1e-3
    Kvj = 1e-3

    # --- Load robot model
    builder = RobotWrapper.BuildFromURDF
    robot = builder(
                    "/home/quant/ros/ocs2_ws/src/HURON-Model/huron_description/urdf/huron_cheat.urdf",
                    ["/home/quant/ros/ocs2_ws/src/HURON-Model/huron_description"],
                    None,
                )
    robot.q0 = np.array([0, 0, 1.0627, 0, 0, 0, 1,
                        0.0000,  0.0000, -0.3207, 0.7572, -0.4365,  0.0000,
                        0.0000,  0.0000, -0.3207, 0.7572, -0.4365,  0.0000])
    
    xc = robot.q0[7:]
    vc = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    # Open the viewer
    viz = MeshcatVisualizer(robot)
    viz.display(robot.q0)

    # The pinocchio model is what we are really interested by.
    model = robot.model
    data = model.createData()

    base_ID = model.getFrameId(baseFrameName)
    endEffector_ID = model.getFrameId(endEffectorFrameName)
    fixedFoot_ID = model.getFrameId(fixedFootFrameName)
    for c in contacts:
        c.id = model.getFrameId(c.name)
        assert c.id < len(model.frames)
        c.jid = model.frames[c.id].parentJoint
        c.placement = model.frames[c.id].placement
        c.model = pin.RigidConstraintModel(c.type, model, c.jid, c.placement)
    contact_models = [c.model for c in contacts]
    
    # Tuning of the proximal solver (minimal version)
    prox_settings = pin.ProximalSettings(0, 1e-6, 1)

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
    ccontact_models = [cpin.RigidConstraintModel(c) for c in contact_models]
    ccontact_datas = [c.createData() for c in ccontact_models]
    cprox_settings = cpin.ProximalSettings(
        prox_settings.absolute_accuracy, prox_settings.mu, prox_settings.max_iter
    )
    cpin.initConstraintDynamics(cmodel, cdata, ccontact_models)

    nq = model.nq # 19
    nv = model.nv # 18
    nx = nq + nv 
    ndx = 2 * nv 
    cx = casadi.SX.sym("x", nx, 1)
    cdx = casadi.SX.sym("dx", ndx, 1)
    cq = cx[:nq] # first 19 of x are q
    cv = cx[nq:] # last 18 of x are v
    caq = casadi.SX.sym("a", nv, 1) # 18 acceleration variables
    ctauq = casadi.SX.sym("tau", nv, 1) # 18 torques

    # Compute kinematics casadi graphs
    cpin.constraintDynamics(cmodel, cdata, cq, cv, ctauq, ccontact_models, ccontact_datas)
    cpin.forwardKinematics(cmodel, cdata, cq, cv, caq)
    cpin.updateFramePlacements(cmodel, cdata)

    # Sym graph for the integration operation x,dx -> x(+)dx = [model.integrate(q,dq),v+dv]
    cintegrate = casadi.Function(
        "integrate",
        [cx, cdx],
        [casadi.vertcat(cpin.integrate(cmodel, cx[:nq], cdx[:nv]), cx[nq:] + cdx[nv:])],
    )

    # Sym graph for the integration operation x' = [ q+vDT+aDT**2, v+aDT ]
    cnext = casadi.Function(
        "next",
        [cx, caq],
        [
            casadi.vertcat(
                cpin.integrate(cmodel, cx[:nq], cx[nq:] * DT + caq * DT**2),
                cx[nq:] + caq * DT,
            )
        ],
    )

    # Sym graph for the aba operation
    caba = casadi.Function("fdyn", [cx, ctauq], [cdata.ddq])

    # Sym graph for the operational error
    error_tool = casadi.Function(
        "etool3", [cx], [Mtarget.translation - cdata.oMf[endEffector_ID].translation]
    )
    
    # Sym graph for the contact constraint and Baugart correction terms
    # Works for both 3D and 6D contacts.
    # Uses the contact list <contacts> where each item must have a <name>, an <id> and a <type> field.
    dpcontacts = {}  # Error in contact position
    vcontacts = {}  # Error in contact velocity
    acontacts = {}  # Contact acceleration

    for c in contacts:
        if c.type == pin.ContactType.CONTACT_3D:
            p0 = data.oMf[c.id].translation.copy()
            dpcontacts[c.name] = casadi.Function(
                f"dpcontact_{c.name}",
                [cx],
                [-(cdata.oMf[c.id].inverse().act(casadi.SX(p0)))],
            )
            vcontacts[c.name] = casadi.Function(
                f"vcontact_{c.name}",
                [cx],
                [cpin.getFrameVelocity(cmodel, cdata, c.id, pin.LOCAL).linear],
            )
            acontacts[c.name] = casadi.Function(
                f"acontact_{c.name}",
                [cx, caq],
                [cpin.getFrameClassicalAcceleration(cmodel, cdata, c.id, pin.LOCAL).linear],
            )
        elif c.type == pin.ContactType.CONTACT_6D:
            p0 = data.oMf[c.id]
            dpcontacts[c.name] = casadi.Function(f"dpcontact_{c.name}", [cx], [np.zeros(6)])
            vcontacts[c.name] = casadi.Function(
                f"vcontact_{c.name}",
                [cx],
                [cpin.getFrameVelocity(cmodel, cdata, c.id, pin.LOCAL).vector],
            )
            acontacts[c.name] = casadi.Function(
                f"acontact_{c.name}",
                [cx, caq],
                [cpin.getFrameAcceleration(cmodel, cdata, c.id, pin.LOCAL).vector],
            )
            
    # Get initial contact position (for Baumgart correction)
    pin.framesForwardKinematics(model, data, robot.q0)
    
    cbaumgart = {
        c.name: casadi.Function(
            f"K_{c.name}", [cx], [200. * dpcontacts[c.name](cx) + 2. * np.sqrt(200.) * vcontacts[c.name](cx)]
        )
        for c in contacts
    }

    ### PROBLEM

    opti = casadi.Opti()
    var_dxs = [opti.variable(ndx) for t in range(T + 1)]
    var_as = [opti.variable(nv) for t in range(T)]
    var_us = [opti.variable(nv - 6) for t in range(T)]
    var_xs = [
        cintegrate(np.concatenate([robot.q0, np.zeros(nv)]), var_dx) for var_dx in var_dxs
    ]

    totalcost = 0
    # Define the running cost
    for t in range(T):
        totalcost += 1e-3 * DT * casadi.sumsqr(var_xs[t][nq:])
        totalcost += 1e-4 * DT * casadi.sumsqr(var_as[t])
    # for t in range(T):
    #     totalcost += 1e-4 * DT * casadi.sumsqr(var_us[t])
    totalcost += 1e4 * casadi.sumsqr(error_tool(var_xs[T]))

    opti.subject_to(var_xs[0][:nq] == robot.q0)
    opti.subject_to(var_xs[0][nq:] == 0)  # zero initial velocity
    # opti.subject_to(var_xs[T][nq:] == 0)  # zero terminal velocity

    # Define the integration constraints
    for t in range(T):
        tau = casadi.vertcat(np.zeros(6), var_us[t])
        opti.subject_to(caba(var_xs[t], tau) == var_as[t])
        opti.subject_to(cnext(var_xs[t], var_as[t]) == var_xs[t + 1])
    
    for t in range(T):
        for c in contacts:
            correction = cbaumgart[c.name](var_xs[t])
            opti.subject_to(acontacts[c.name](var_xs[t], var_as[t]) == -correction)

    ### SOLVE
    opti.minimize(totalcost)
    p_opts = {"expand": True}
    s_opts = {"max_iter": 25}
    opti.solver("ipopt", p_opts, s_opts) # set numerical backend
    opti.callback(lambda i: displayScene(opti.debug.value(var_xs[-1][:nq])))

    # Caution: in case the solver does not converge, we are picking the candidate values
    # at the last iteration in opti.debug, and they are NO guarantee of what they mean.
    try:
        sol = opti.solve_limited()
        sol_xs = [opti.value(var_x) for var_x in var_xs]
        sol_us = [opti.value(var_u) for var_u in var_us]
    except:
        print("ERROR in convergence, plotting debug info.")
        sol_xs = [opti.debug.value(var_x) for var_x in var_xs]
        sol_us = [opti.debug.value(var_u) for var_u in var_us]

    print("***** Display the resulting trajectory ...")

    xdes = [x[:nq] for x in sol_xs]
    vdes = [x[nq:nq+nv] for x in sol_xs]
    taudes = [u for u in sol_us]
    
    pf1des = []
    pf2des = []
    vf1des = []
    vf2des = []
    
    for t in range(T):
        pin.forwardKinematics(model, data, xdes[t], vdes[t])
        pin.framesForwardKinematics(model, data, xdes[t])
        vel1 = pin.getFrameVelocity(model, data, fixedFoot_ID, pin.LOCAL).vector
        vel2 = pin.getFrameVelocity(model, data, endEffector_ID, pin.LOCAL).vector
        
        M1 = data.oMf[fixedFoot_ID]
        M2 = data.oMf[endEffector_ID]
        
        pf1des.append(M1)
        pf2des.append(M2)
        vf1des.append(vel1)
        vf2des.append(vel2)
    # print(pf1des)
    # print(pf2des)
    # print(vf1des)
    # print(vf2des)
    
    # while True:
    #     # displayScene(robot.q0, 1)
    #     displayTraj(xdes, DT)
    #     xdes.reverse()
    #     displayTraj(xdes, DT)
    #     xdes.reverse()
    
    unpause_physics_client(EmptyRequest())
    for t in range(T):
        ms = get_model_state_client(GetModelStateRequest("huron", "ground_plane"))
        p = ms.pose.position
        o = ms.pose.orientation
        vl = ms.twist.linear
        al = ms.twist.angular
        xcb = np.hstack(([p.x, p.y, p.z, o.x, o.y, o.z, o.w], xc))
        vcb = np.hstack(([vl.x, vl.y, vl.z, al.x, al.y, al.z], vc))
        # print(xcb)
        # print(vcb)

        pin.forwardKinematics(model, data, xcb, vcb)
        pin.framesForwardKinematics(model, data, xcb)
        vel1 = pin.getFrameVelocity(model, data, fixedFoot_ID, pin.LOCAL)
        vel2 = pin.getFrameVelocity(model, data, endEffector_ID, pin.LOCAL)
        # print(vel1.vector)
        
        J1 = pin.computeFrameJacobian(model, data, xcb, fixedFoot_ID, pin.LOCAL)
        J2 = pin.computeFrameJacobian(model, data, xcb, endEffector_ID, pin.LOCAL)
        # print(J1)
        # print(J2)
        
        M1 = data.oMf[fixedFoot_ID]
        M2 = data.oMf[endEffector_ID]
        # print(M2)
        # print(M2)
        
        epf = np.hstack((pin.log(M1.inverse() * pf1des[t]).vector, pin.log(M2.inverse() * pf2des[t]).vector))
        print(epf)
        evf = np.hstack((vf1des[t] - vel1.vector, vf2des[t] - vel2.vector))
        # print(evf)
        J1 = np.hstack((J1[:, 6:12], np.zeros((6,6))))
        J2 = np.hstack((np.zeros((6,6)), J2[:, 12:]))
        J = np.vstack((J1, J2))
        # print(J)
        tauff = np.reshape(taudes[t], (12, 1)) + np.matmul(J.transpose(), (np.reshape(Kp * epf, (12, 1)) + np.reshape(Kv * evf, (12, 1))))
        print(tauff)

        tstart = rospy.Time.now().to_sec()
        tnow = tstart
        while tnow < tstart + DT:
            tau = tauff + Kpj * (np.reshape(xdes[t][7:], (12, 1)) - np.reshape(xc, (12, 1))) + Kvj * (np.reshape(vdes[t][6:], (12, 1)) - np.reshape(vc, (12, 1)))
            # print(tau)
            effortController.publish(Float64MultiArray(data=list(tau)))
            tnow = rospy.Time.now().to_sec()
    pause_physics_client(EmptyRequest())

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass