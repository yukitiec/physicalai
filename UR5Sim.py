import os
import math 
import numpy as np
import time
import pybullet 
import random
from datetime import datetime
import pybullet_data
from collections import namedtuple
from addict import Dict

ROBOT_URDF_PATH = "./ur_e_description/urdf/ur5e.urdf"
TABLE_URDF_PATH = os.path.join(pybullet_data.getDataPath(), "table/table.urdf")

class UR5Sim():
  
    def __init__(self, camera_attached=False):
        pybullet.connect(pybullet.GUI)
        pybullet.setRealTimeSimulation(True)
        pybullet.setGravity(0, 0, -9.8)
        
        self.end_effector_index = 7
        self.ur5 = self.load_robot()
        self.num_joints = pybullet.getNumJoints(self.ur5)
        
        self.control_joints = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]
        self.joint_type_list = ["REVOLUTE", "PRISMATIC", "SPHERICAL", "PLANAR", "FIXED"]
        self.joint_info = namedtuple("jointInfo", ["id", "name", "type", "lowerLimit", "upperLimit", "maxForce", "maxVelocity", "controllable"])

        #long rope
        self.rope_length = 2.0  # Length of the rope in meters
        self.num_links = 20  # Number of links in the rope
        self.link_length = self.rope_length / self.num_links
        self.link_mass = 0.1  # Mass of each link
        self.rope_joints = []

        #turn around the rope.
        self.moving_point_radius = 0.3  # Radius of the vertical circular path
        self.moving_point_horizontal_offset_x = 0.6  # Horizontal distance away from the robot
        self.moving_point_horizontal_offset_y = 0.6  # Horizontal distance away from the robot
        self.moving_point_center = [self.moving_point_horizontal_offset_x, self.moving_point_horizontal_offset_x, 1.0]  # Initial center point
        self.moving_point_speed = 4.0  # Speed of the circular motion

        # Load the rope
        self.load_rope()

        self.joints = Dict()
        for i in range(self.num_joints):
            info = pybullet.getJointInfo(self.ur5, i)
            jointID = info[0]
            jointName = info[1].decode("utf-8")
            jointType = self.joint_type_list[info[2]]
            jointLowerLimit = info[8]
            jointUpperLimit = info[9]
            jointMaxForce = info[10]
            jointMaxVelocity = info[11]
            controllable = True if jointName in self.control_joints else False
            info = self.joint_info(jointID, jointName, jointType, jointLowerLimit, jointUpperLimit, jointMaxForce, jointMaxVelocity, controllable)
            if info.type == "REVOLUTE":
                pybullet.setJointMotorControl2(self.ur5, info.id, pybullet.VELOCITY_CONTROL, targetVelocity=0, force=0)
            self.joints[info.name] = info     


    def load_robot(self):
        flags = pybullet.URDF_USE_SELF_COLLISION
        table = pybullet.loadURDF(TABLE_URDF_PATH, [0.5, 0, -0.6300], [0, 0, 0, 1])
        robot = pybullet.loadURDF(ROBOT_URDF_PATH, [0, 0, 0], [0, 0, 0, 1], flags=flags)
        return robot
    
    def load_rope(self):
        pybullet.setAdditionalSearchPath(pybullet_data.getDataPath())  # Load the URDF files
        link_visual_shape_id = pybullet.createVisualShape(pybullet.GEOM_CYLINDER, 
                                                          radius=0.01, 
                                                          length=self.link_length,
                                                          rgbaColor=[0.0, 0.0, 1.0, 1.0])

        # Create the first link attached to the end-effector
        link_collision_shape_id = pybullet.createCollisionShape(pybullet.GEOM_CYLINDER, 
                                                                radius=0.01, 
                                                                height=self.link_length)
        ur5_end_effector_pos, _ = pybullet.getLinkState(self.ur5, self.end_effector_index)[0:2]
        prev_link_id = pybullet.createMultiBody(baseMass=self.link_mass,
                                                baseCollisionShapeIndex=link_collision_shape_id,
                                                baseVisualShapeIndex=link_visual_shape_id,
                                                basePosition=ur5_end_effector_pos)

        # Attach the first link to the UR5 end-effector using a fixed joint
        pybullet.createConstraint(self.ur5, self.end_effector_index, 
                                  prev_link_id, -1, 
                                  pybullet.JOINT_FIXED, 
                                  jointAxis=[0, 0, 0],
                                  parentFramePosition=[0, 0, 0],
                                  childFramePosition=[0, 0, self.link_length / 2])

        # Create the remaining rope links
        for i in range(1, self.num_links):
            link_id = pybullet.createMultiBody(baseMass=self.link_mass,
                                               baseCollisionShapeIndex=link_collision_shape_id,
                                               baseVisualShapeIndex=link_visual_shape_id,
                                               basePosition=[ur5_end_effector_pos[0], ur5_end_effector_pos[1], ur5_end_effector_pos[2] - i * self.link_length])
            
            # Create a hinge or spherical joint to connect the previous link to the current one
            pybullet.createConstraint(prev_link_id, -1, link_id, -1, 
                                      pybullet.JOINT_POINT2POINT, 
                                      jointAxis=[0, 0, 0],
                                      parentFramePosition=[0, 0, -self.link_length / 2],
                                      childFramePosition=[0, 0, self.link_length / 2])
            
            prev_link_id = link_id  # Update for the next iteration
            self.rope_joints.append(link_id)

        # Attach the last link to the moving point
        self.last_link_id = prev_link_id
        self.create_moving_point()


    def set_joint_angles(self, joint_angles):
        poses = []
        indexes = []
        forces = []

        for i, name in enumerate(self.control_joints):
            joint = self.joints[name]
            poses.append(joint_angles[i])
            indexes.append(joint.id)
            forces.append(joint.maxForce)

        pybullet.setJointMotorControlArray(
            self.ur5, indexes,
            pybullet.POSITION_CONTROL,
            targetPositions=joint_angles,
            targetVelocities=[0]*len(poses),
            positionGains=[0.04]*len(poses), forces=forces
        )


    def get_joint_angles(self):
        j = pybullet.getJointStates(self.ur5, [1,2,3,4,5,6])
        joints = [i[0] for i in j]
        return joints
    

    def check_collisions(self):
        collisions = pybullet.getContactPoints()
        if len(collisions) > 0:
            print("[Collision detected!] {}".format(datetime.now()))
            return True
        return False


    def calculate_ik(self, position, orientation):
        quaternion = pybullet.getQuaternionFromEuler(orientation)
        lower_limits = [-math.pi]*6
        upper_limits = [math.pi]*6
        joint_ranges = [2*math.pi]*6
        rest_poses = [0, -math.pi/2, -math.pi/2, -math.pi/2, -math.pi/2, 0]

        joint_angles = pybullet.calculateInverseKinematics(
            self.ur5, self.end_effector_index, position, quaternion, 
            jointDamping=[0.01]*6, upperLimits=upper_limits, 
            lowerLimits=lower_limits, jointRanges=joint_ranges, 
            restPoses=rest_poses
        )

        return joint_angles
    
    def end_effector_velocity_to_joint_velocity(self, end_effector_linear_vel, end_effector_angular_vel):
        # Get the Jacobian
        zero_vec = [0.0] * pybullet.getNumJoints(self.ur5)
        
        # position and orientation of end-effector (to get Jacobian at the current state)
        link_state = pybullet.getLinkState(self.ur5, self.end_effector_index, computeLinkVelocity=1, computeForwardKinematics=1)
        current_joint_angles = pybullet.calculateInverseKinematics(self.ur5, self.end_effector_index, link_state[4], link_state[5])

        # Calculate the Jacobian for the end-effector
        jac_t, jac_r = pybullet.calculateJacobian(self.ur5, self.end_effector_index, [0, 0, 0],
                                                current_joint_angles, zero_vec, zero_vec)

        # Convert the Jacobians from tuple format to numpy array
        jac_t = np.array(jac_t)
        jac_r = np.array(jac_r)

        # Combine linear and angular velocities into a single vector
        end_effector_vel = np.hstack((end_effector_linear_vel, end_effector_angular_vel))

        # Combine the translational and rotational Jacobian matrices
        jacobian = np.vstack((jac_t, jac_r))

        # Compute joint velocities by multiplying Jacobian pseudo-inverse with end-effector velocity
        joint_velocities = np.linalg.pinv(jacobian).dot(end_effector_vel)

    def add_gui_sliders(self):
        self.sliders = []
        self.sliders.append(pybullet.addUserDebugParameter("X", 0, 1, 0.4))
        self.sliders.append(pybullet.addUserDebugParameter("Y", -1, 1, 0))
        self.sliders.append(pybullet.addUserDebugParameter("Z", 0.3, 1, 0.4))
        self.sliders.append(pybullet.addUserDebugParameter("Rx", -math.pi/2, math.pi/2, 0))
        self.sliders.append(pybullet.addUserDebugParameter("Ry", -math.pi/2, math.pi/2, 0))
        self.sliders.append(pybullet.addUserDebugParameter("Rz", -math.pi/2, math.pi/2, 0))


    def read_gui_sliders(self):
        x = pybullet.readUserDebugParameter(self.sliders[0])
        y = pybullet.readUserDebugParameter(self.sliders[1])
        z = pybullet.readUserDebugParameter(self.sliders[2])
        Rx = pybullet.readUserDebugParameter(self.sliders[3])
        Ry = pybullet.readUserDebugParameter(self.sliders[4])
        Rz = pybullet.readUserDebugParameter(self.sliders[5])
        return [x, y, z, Rx, Ry, Rz]
        
    def get_current_pose(self):
        linkstate = pybullet.getLinkState(self.ur5, self.end_effector_index, computeForwardKinematics=True)
        position, orientation = linkstate[0], linkstate[1]
        return (position, orientation)
    
    def create_moving_point(self):
        # Create a massless point at the moving point's initial position
        self.point_id = pybullet.createMultiBody(baseMass=0,  # Massless point
                                                 baseCollisionShapeIndex=-1,
                                                 basePosition=self.moving_point_center)

        # Attach the last rope link to the point with a fixed joint
        pybullet.createConstraint(self.last_link_id, -1, self.point_id, -1,
                                  pybullet.JOINT_FIXED,
                                  jointAxis=[0, 0, 0],
                                  parentFramePosition=[0, 0, -self.link_length / 2],
                                  childFramePosition=[0, 0, 0])

    def update_moving_point(self, time_step):
        # Calculate the new position of the moving point in a vertical circular path
        z = self.moving_point_center[2] + self.moving_point_radius * math.sin(self.moving_point_speed * time_step)
        y = self.moving_point_center[1] + self.moving_point_radius * math.cos(self.moving_point_speed * time_step)
        
        # Only the x-coordinate has a fixed horizontal offset from the base by 0.7 meters
        x = self.moving_point_horizontal_offset_x
        y = self.moving_point_horizontal_offset_y

        # Update the position of the moving point
        pybullet.resetBasePositionAndOrientation(self.point_id, [x, y, z], [0, 0, 0, 1])

def demo_simulation():
    """ Demo program showing how to use the sim """
    if pybullet.isConnected():
        pybullet.disconnect()
    # pybullet.resetDebugVisualizerCamera(
    #     cameraDistance=1.5,           # Distance of the camera from the object
    #     cameraYaw=50,                 # Yaw angle (horizontal rotation around the object)
    #     cameraPitch=-35,              # Pitch angle (vertical rotation around the object)
    #     cameraTargetPosition=[0, 0, 1]  # Target position (where the camera is pointing)
    # )
    sim = UR5Sim()
    sim.add_gui_sliders()

    # Simulation time step
    time_step = 0
    while True:
        pybullet.stepSimulation()
        sim.update_moving_point(time_step)
        time_step += 0.01
        time.sleep(1/240)
    # while True:
    #     x, y, z, Rx, Ry, Rz = sim.read_gui_sliders()
    #     joint_angles = sim.calculate_ik([x, y, z], [Rx, Ry, Rz])
    #     sim.set_joint_angles(joint_angles)
    #     sim.check_collisions()



if __name__ == "__main__":
    demo_simulation()