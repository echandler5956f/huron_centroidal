#!/home/quant/ros_ws/src/huron_centroidal/.venv/bin/python3

import rospy
import time
import matplotlib.pyplot as plt
from types import SimpleNamespace

import numpy as np
import pinocchio as pin
from pinocchio.robot_wrapper import RobotWrapper

from utils.meshcat_viewer_wrapper import MeshcatVisualizer

import huron_centroidal_v2 as hc

def main():
     # Change numerical print
    pin.SE3.__repr__ = pin.SE3.__str__
    np.set_printoptions(precision=2, linewidth=300, suppress=True, threshold=1e6)

    ### HYPER PARAMETERS
    Mtarget = pin.SE3(pin.utils.rotate("y", np.pi/2), np.array([0.0775, 0.05, 0.1]))  # x,y,z
    contacts = [SimpleNamespace(name="l_foot_v_ft_link", type=pin.ContactType.CONTACT_6D)]
    baseFrameName = "base"
    endEffectorFrameName = "r_foot_v_ft_link"
    fixedFootFrameName = "l_foot_v_ft_link"

    # cons = hc.ContactSequence([hc.ContactPhase([hc.Phase(), hc.Phase()], ))
    # trajpt = hc.CentroidalTrajOpt()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass