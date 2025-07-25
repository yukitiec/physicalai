import pybullet as p
import pybullet_data
import numpy as np
import gym
from gym import spaces
import time
import random
from gym.utils import seeding
import math
import cv2
import matplotlib.pyplot as plt
import os

import ur_custom as ur_ctrl
import robot_driver as ur_driver


class UR5eRopeEnv(gym.Env):
    def __init__(self, fps,step_episode,client_id):
        super(UR5eRopeEnv, self).__init__()
        self.client_id=client_id
        self._bool_debug = False
        self.state_id = 0 #For saving the state. 
        self.step_episode = step_episode
        self.weight_speed = 0.05 #weight for speed

        """Normalizer"""
        #workspace
        self._normalizer={
            "p_min":[0,0,0],
            "p_max":[5.0,5.0,5.0],
            "q_min":-2.0*np.pi,
            "q_max":2.0*np.pi,
            "omega_min":-np.pi,
            "omega_max":np.pi,
            "v_min":0.0,
            "v_max":5.0,
            "force_min":0.0,
            "force_max":50.0,
            "torque_min":0.0,
            "torque_max":10.0,
            "l_rope_max":5.0,
        }
        #X
        self._x_min = -4.0 #m
        self._x_max = 1.0 #m
        #Y
        self._y_min = -2.5 #m
        self._y_max = 2.5 #m
        #Z
        self._z_min = -1.0 #m
        self._z_max = 4.0 #m
        #position.
        #self._p_min = [self._x_min,self._y_min,self._z_min]
        #self._p_max = [self._x_max,self._y_max,self._z_max]
        #normalize only the norm
        self._p_min = self._normalizer["p_min"]
        self._p_max = self._normalizer["p_max"]

        #orientation. rotation vector.
        self._q_min = self._normalizer["q_min"]
        self._q_max = self._normalizer["q_max"]
        #angular velocity.
        self._omega_min = self._normalizer["omega_min"]
        self._omega_max = self._normalizer["omega_max"]

        #speed range
        self._v_min = self._normalizer["v_min"] #m/s
        self._v_max = self._normalizer["v_max"] #m/s

        #force range
        self._force_min = self._normalizer["force_min"] #N
        self._force_max = self._normalizer["force_max"] #N
        self._torque_min = self._normalizer["torque_min"] #Nm
        self._torque_max = self._normalizer["torque_max"] #Nm

        #rope 
        self._l_rope_max = self._normalizer["l_rope_max"] #m. rope length normalizer.
        """"""
        
        """CHECK HERE"""
        self.idx_reward = 3 #0: geometry-based reward. 1: imitation-based reward.2:mixed, 3: long-term simple reward on 20250603.
        self.alpha_imitate = 0.9 #ratio of imitation-based reward in the total one.
        
        # Rope setting.
        #mass. general long rope mass : 5.67e-4 kg/cm -> 30m-1.7 kg
        #30m-0.6 kg~3.6 kg.
        self._m_rope_lb = 2.0e-4 #Upper bound for the link mass
        self._m_rope_ub = 1.2e-3 #upper bound for the link mass
        #length
        self._l_rope_lb = 0.8 #Upper bound for the rope length
        self._l_rope_ub = 3.0 #upper bound for the rope length

        #Human motion setting.
        self._rate_h2r_lb = 0.65
        self._rate_h2r_ub = 0.8
        self._z_lb = 0.4 #lower bound for the human's initial position
        self._z_ub = 0.9 #upper bound for the human's initial position
        self._r_min_turn = 0.15 #lower bound for the human's turning radius.
        self._r_max_turn = 0.5 #upper bound for the human's turning radius.
        self._zmin_rob=self._z_lb-self._l_rope_ub*self._r_min_turn-0.1 #minimum Z for the robot.
        self._zmax_rob=self._z_ub+self._l_rope_ub*self._r_min_turn+0.1 #maximum z for the robot.
        self._v_hum_ub=10.0 #upper bound for the human speed. 15 m/s -> 48 km/h
        self._v_hum_lb = 0.5 #lower bound for the human speed. 0.5 m/s -> 1.6 km/h
        self._omega_hum_ub = 4.0*math.pi #upper bound for the human's angular velocity.
        self._omega_hum_lb = 2.0*math.pi #lower bound for the human's angular velocity.
        self._n_max_turn = 1.5 #max turning circuits per second. 5 circuits per second.
        #change human motion
        self._n_circuit_update = 10 #10 circuits. How often do we update the turning conditions in rope turning?

        #human wrist&s motion.
        self._idx_human_motion = 1
        #1. epsilon-greedy method.
        self._epsilon_motion = 0.1 #0.9 : ordinary speed. 0.1 : abnormal speed.
        self._variance_motion = 0.1 #+- 0.1*velocity.

        #2. downward : accelerate, upward : deccelerate.
        self._variance_acc = 0.1 #acceleration variaon. +-0.1
        
        #Robot's motion setting.###
        self._v_max_eef = 1.0 #1.0 m/s
        self._d_deccelerate = 0.05 #deccelerating zone. 0.05 m around the target position
        self._omega_max_eef = math.pi #pi rad/s
        self._theta_deccelerate = math.pi/10.0 # deccelerating zone. pi/10 rad around the target pose.
        self._epsilon_noize = 0.0#0.01 #human's motion noise. 1 cm order.

        #reward setting
        self._penalty_length = -2.0 #penalty for distance  
        self._w_penalty_convex = 0.5 #weight for convexity-based reward against amplitude-based reward.
        self._w_penalty_length = 5.0 #weight for rope's length limitations. 5.0 against the amplitude
        self._unit_length = 0.05 # [m] minimal unit for length-aware reward. use in convexity-based reward.
        ###########################

        #storage
        self.samples_turn = np.array([[0,0,0]])#(360,3(x,y,z))
        self.vel_samples_turn = []#(360,) : How many steps to proceed if the wrist is in the correspondent 3D position.
        self.deg_per_sec = 0
        self.R_r2h=0
        self.n_circuit = 0 #How many circuits
        self.n_degree = 0 #How many degree the human's wrist proceeds.
        self.pos_human_current = np.array([0,0,0])#(3,): current human's wrist position. [x,y,z]
        self.pos_human_prev = np.array([0,0,0])#(3,)
        self.vel_human = np.array([0,0,0])#(3,). human's wrist speed. [vx,vy,vz]
        self.dist_r2h_current = 0
        self.target_robot=1e5*np.zeros((3,3)) #for saving target positions.

        self.r_current=-1 
        self.r_previous=-1 
        self.center_prev = np.zeros(4) #previous fitting center. (3)
        self.middle_rope_list = [] #storage for saving middle points
        self._max_size=100 #maximum storage.

        self.ur_pose_env=[]
        self.ur_pose_custom = []
        self.sphere_ids = []
        self.cylinder_ids = []

        #robot's configuration.
        self.config_robot = np.zeros((6,6))#6 joints' 6D pose. (x,y,z,rx,ry,rz)
        self.pose_eef = np.zeros(6) #6D eef pose.
        self.joints_list = np.zeros(6) #robot's joint's angles. [rad]
        self.omega_joint = np.zeros(6) #joint's angular velocity.
        self.v_eef = np.zeros(6) #eef velocity.
        self.sphere_radius = 0.0755  # 10 cm
        self.cylinder_radius = 0.07  # 10 cm
        self.joint_storage = []

        """CHECK HERE"""
        # Initialize PyBullet environment.
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.client_id)
        p.setGravity(0, 0, -9.8, physicsClientId=self.client_id)

        # Load plane at default position (0,0,0)
        self.PLANE = r"../plane/plane.urdf"#"C:/Users/kawaw/python/pybullet/ur5-bullet/UR5/code/robot_RL_curriculum/plane/plane.urdf"
        self.plane = p.loadURDF(self.PLANE, physicsClientId=self.client_id)

        #set frictino
        p.changeDynamics(
            self.plane,
            linkIndex=-1,
            lateralFriction=0.5, #default is 0.5
            spinningFriction=0.001,
            rollingFriction=0.001,
            physicsClientId=self.client_id
        )

        #robot
        self._ur_ctrl = ur_ctrl.UrCustom() #get the instance of UR kinematics.
        self._ur_driver = ur_driver.RobotDriver()
        
        #control Frequency
        self.frequency = fps #10 Hz or 500 Hz
        self.dt_sample = 1.0/self.frequency #sampling time. [s]
        p.setTimeStep(self.dt_sample, physicsClientId=self.client_id)  # Set the control frequency.
        
        # Target movement parameters
        self.moving_point_radius = 0.20  # Vertical radius
        self.moving_point_horizontal_offset_x = -0.1  #-0.5 # Horizontal offset (x-axis)
        self.moving_point_horizontal_offset_y = 0.2 #0.5
        self.moving_point_center = [self.moving_point_horizontal_offset_x, self.moving_point_horizontal_offset_y, 0.55] #0.6
        self.target_position = np.array([self.moving_point_center[0] + self.moving_point_radius, self.moving_point_horizontal_offset_y, self.moving_point_center[2]])
        self.target_speed = 2.0*math.pi  # Will be randomized per episode
        self.target_angle = 0.0
        
        # Define maximum steps per episode
        self.max_episode_steps = self.frequency*20.0  # e.g. 20 seconds at self.frequency Hz
        
        # Rope parameters
        self.rope_length = 0.5 #0.4 m -> 0.5 m
        self.num_links = 50
        self.link_length = self.rope_length / self.num_links
        self.link_mass = 0.01 #kg/m -> 1.0
        self.rope_link_ids = []
        self.counter_save_robot = 0
        self.prev_link_vel = 1e5 #previous link velocity. This is for estimating force and torque.
        self.rope_mid_point = np.zeros(3)
        self.prev_link_vels = {}
        #camera setting
        # Camera parameters
        self.camera_distance =1.2  # Distance from the target (3 meters)
        self.camera_pitch = 0.0     # Keep pitch at 0 degrees
        self.camera_yaw = 0.0       # Start yaw at 0 degrees
        self.camera_target = [0, 0, 0]  # Target point in world coordinates
        self.x_cam,self.y_cam,self.z_cam = 0,0,0
        self.p_cam_goal = np.zeros(3)

        # Action space: next target position
        self.action_space = spaces.Box(low=-1, high=1, shape=(6,), dtype=np.float32) #(x,y,z,rx,ry,rz) #adopt the rotation vector. q.x,q.y,q.z,q.w)

        # Observation space: [eef_pose(6),eef_vel(6), pos_human(3),vel_human(3),force_eef(3),torque_eef(3),rope_length(1), p_middle(3),vel_middle(3)]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(31,), dtype=np.float32)
        
        #for replay
        self.rosbag={"x_cam":0,"y_cam":0,"z_cam":0,"target_position":np.zeros((2,3)),"lateralFriction":0,
                 "spinningFriction":0,"rolliingFriction":0,"link_mass":0,"rope_length":0,"link_length":0,"num_links":0} #dictionary for saving the data.
        self.bool_base = False #For training (False), or baseline (True).
        self.counter_step = 0

        # Initialize state variables
        self.time_step = 1
        #self.reset()

        self.bool_eval = False

    def seed(self, seed=None):
        """Sets the seed for the environment and randomness."""
        #if seed is None:
        #    seed = np.random.randint(0, 2**32 - 1)
        
        #np.random.seed(seed)  # Seed NumPy's random generator
        #random.seed(seed)     # Seed Python's random module
        
        #self.np_random, seed = seeding.np_random(seed)  # Optionally, for OpenAI Gym seeding compatibility
        
        return seed
    
    def load_rope(self):
        """Initialize the rope as a chain of links attached to the UR5e's end-effector.
        """
        self.rope_link_ids = []
        link_visual_shape_id = p.createVisualShape(p.GEOM_CYLINDER, 
                                                  radius=0.03, 
                                                  length=self.link_length,
                                                  rgbaColor=[1.0, 0.0, 0.0, 1.0], 
                                                  physicsClientId=self.client_id)  # Blue color
        
        link_collision_shape_id = p.createCollisionShape(p.GEOM_CYLINDER, 
                                                        radius=0.03, 
                                                        height=self.link_length, 
                                                        physicsClientId=self.client_id)
        
        prev_link_id = p.createMultiBody(baseMass=self.link_mass,
                                baseCollisionShapeIndex=link_collision_shape_id,
                                baseVisualShapeIndex=link_visual_shape_id,
                                basePosition=self.pose_eef[:3]#3D robot's joint.
                                , physicsClientId=self.client_id
                        )
        
        # Attach the last rope link to the moving point
        self.root_link_id = prev_link_id
        self.create_moving_point_root()
        
        self.rope_link_ids.append(prev_link_id)
        
        p_end = self.target_position
        p_st = self.pose_eef[:3]
        horizontal_vec = p_end - p_st
        horizontal_dist = np.linalg.norm(horizontal_vec)
        # Normalize horizontal direction
        dir_vec = horizontal_vec / horizontal_dist
        dp = (p_end-p_st)/self.num_links#np.linalg.norm(p_end-p_st)
        sag_amplitude = self.rope_length * 0.2  # Tune this as needed
        
        # Create the remaining rope links
        for i in range(1, self.num_links):
            t = i / self.num_links
            
            link_pos = self.get_rope_point(p_st=p_st,dir_vec=dir_vec, horizontal_dist=horizontal_dist,
                                      sag_amplitude=sag_amplitude,t=t)
            
            link_id = p.createMultiBody(baseMass=self.link_mass,
                                        baseCollisionShapeIndex=link_collision_shape_id,
                                        baseVisualShapeIndex=link_visual_shape_id,
                                        basePosition=link_pos, 
                                        physicsClientId=self.client_id)
            
            p.createConstraint(parentBodyUniqueId=prev_link_id,
                               parentLinkIndex=-1,
                               childBodyUniqueId=link_id,
                               childLinkIndex=-1,
                               jointType=p.JOINT_POINT2POINT,
                               jointAxis=[0,0,0],
                               parentFramePosition=[0,0,-self.link_length / 2],
                               childFramePosition=[0,0, self.link_length / 2], 
                               physicsClientId=self.client_id)
            
            self.rope_link_ids.append(link_id)
            prev_link_id = link_id
        
        # Attach the last rope link to the moving point
        self.last_link_id = prev_link_id
        self.create_moving_point()
    
    # Create sagging shape — use sine or parabola for initial guess
    def get_rope_point(self, p_st,dir_vec, horizontal_dist,sag_amplitude,t):
        
        # t ∈ [0, 1]
        pos = p_st + dir_vec * (t * horizontal_dist)
        pos[2] -= sag_amplitude * np.sin(np.pi * t)  # sinusoidal sag

        return pos

    def create_moving_point_root(self):
        """Create a massless point that the first rope link is attached to.
        This point should follow the robot's end-effector position.
        """

        self.point_id_root = p.createMultiBody(baseMass=0,
                                         baseCollisionShapeIndex=-1,
                                         basePosition=self.pose_eef[:3],
                                         physicsClientId=self.client_id
                                         )
        
        # Attach the first rope link to the massless point
        p.createConstraint(parentBodyUniqueId=self.root_link_id,
                           parentLinkIndex=-1,
                           childBodyUniqueId=self.point_id_root,
                           childLinkIndex=-1,
                           jointType=p.JOINT_FIXED,  # Fixed joint to keep them together
                           jointAxis=[0,0,0],
                           parentFramePosition=[0,0,-self.link_length / 2],
                           childFramePosition=[0,0,0],
                           physicsClientId=self.client_id
                           )
        
        # Note: The massless point will be updated to follow the robot's end-effector
        # in the update_target_position_root() method

    def create_moving_point(self):
        """Create a massless point that the last rope link is attached to."""
        sphere_visual = p.createVisualShape(
            shapeType=p.GEOM_SPHERE,
            radius=self.sphere_radius,
            rgbaColor=[0.0, 1.0, 0.0, 1.0], #lightblue
            physicsClientId=self.client_id
        )

        self.point_id = p.createMultiBody(baseMass=0,
                                         baseCollisionShapeIndex=-1,
                                         baseVisualShapeIndex=sphere_visual,
                                         basePosition=self.target_position,
                                           physicsClientId=self.client_id)
        
        # Attach the last rope link to the moving point
        p.createConstraint(parentBodyUniqueId=self.last_link_id,
                           parentLinkIndex=-1,
                           childBodyUniqueId=self.point_id,
                           childLinkIndex=-1,
                           jointType=p.JOINT_FIXED,  # Allow the link to move #p.JOINT_FIXED,
                           jointAxis=[0,0,0],
                           parentFramePosition=[0,0,-self.link_length / 2],
                           childFramePosition=[0,0,0], 
                           physicsClientId=self.client_id)
    
    def update_target_position_root(self):
        """Update the massless point position to follow the robot's end-effector."""
        # Update the massless point's position to follow the robot's end-effector
        p.resetBasePositionAndOrientation(self.point_id_root, self.pose_eef[:3], [0, 0, 0, 1], physicsClientId=self.client_id)
        #p.resetBasePositionAndOrientation(self.root_link_id, self.pose_eef[:3], [0, 0, 0, 1],physicsClientId=self.client_id)
    
    def update_target_position(self, dt):
        """Update the target position based on current speed and time step."""

        #self.samples_turn :human target position : sample point (360,3(=x,y,z)).
        #self.deg_per_sec : human's turning speed [degree/s]
        #how many degrees the human proceed.
        deg_next = self.n_degree

        self.pos_human_prev = self.target_position.copy() #save the previous human's wrist position.

        if self._idx_human_motion==1:#epsilon-greedy speed.
            if np.random.random()>self._epsilon_motion:#normal motion
                deg_next = int(deg_next+max(min(self.deg_per_sec*dt,359),1))%360
            else:#exploration.
                step_progress = max(1,min(359,np.random.uniform(self.deg_per_sec*dt*(1-self._variance_motion),self.deg_per_sec*dt*(1+self._variance_motion)))) #(1+-variance)    
                deg_next = int(deg_next+step_progress)%360
            
            #update the index.
            if deg_next<self.n_degree:#one circuit.
                self.n_circuit +=1
            self.n_degree = deg_next

            #determine the next position. predefined samples, and noise.
            self.target_position = self.samples_turn[self.n_degree] + np.array([self._epsilon_noize*np.random.random(),self._epsilon_noize*np.random.random(),self._epsilon_noize*np.random.random()])
        elif self._idx_human_motion==2:#upward : deccelerate, downward : accelerate.
            #determine the speed from the predefined samples according to the current position.
            step_progress = max(1,min(359,self.vel_samples_turn[self.n_degree]*dt))#step per frame.
            deg_next = int(deg_next+step_progress)%360
            #update the index.
            if deg_next<self.n_degree:#one circuit.
                self.n_circuit +=1
            self.n_degree = deg_next

            #determine the next position.
            self.target_position = self.samples_turn[self.n_degree] + np.array([self._epsilon_noize*np.random.random(),self._epsilon_noize*np.random.random(),self._epsilon_noize*np.random.random()])

        self.pos_human_current = self.target_position.copy()
        self.vel_human = (self.pos_human_current-self.pos_human_prev)/dt #human's wrist's velocity. [m/s]

        if self.bool_eval==False and self.n_circuit%self._n_circuit_update==0:#we can update the human's turning speed.
            self.changeMotion() #change motion.
        
        #add the action to the rosbag.
        if self.bool_base:#baseline model.
            self.target_position = self.rosbag["target_position"][self.counter_step%self.rosbag["target_position"].shape[0]]
            self.counter_step += 1
        else:
            self.rosbag["target_position"] = np.vstack((self.rosbag["target_position"],np.array([self.target_position])))
        
        # Update the moving point's position in the simulation
        p.resetBasePositionAndOrientation(self.point_id, self.target_position, [0, 0, 0, 1], 
                                          physicsClientId=self.client_id) #(bodyUniqueID, posObj, oriObj(=quaternion))
        
    def make_robot_skeleton(self, joint_positions):
        self.sphere_ids = []
        self.cylinder_ids = []

        base_position = [0, 0, 0]
        points = np.vstack((np.array([base_position]), joint_positions[:, :3]))

        # Create visual and collision shapes for spheres
        sphere_collision = p.createCollisionShape(
            shapeType=p.GEOM_SPHERE,
            radius=self.sphere_radius,
            physicsClientId=self.client_id
        )
        sphere_visual = p.createVisualShape(
            shapeType=p.GEOM_SPHERE,
            radius=self.sphere_radius,
            rgbaColor=[0.9, 0.84, 0.67, 1.0],
            physicsClientId=self.client_id
        )
        eef_visual = p.createVisualShape(
            shapeType=p.GEOM_SPHERE,
            radius=self.sphere_radius,
            rgbaColor=[0.9, 0.84, 0.67, 1.0],
            physicsClientId=self.client_id
        )

        # Create spheres with collision
        for i, pos in enumerate(points):
            visual = sphere_visual if i < joint_positions.shape[0] - 1 else eef_visual
            sphere_id = p.createMultiBody(
                baseMass=0.0,  # static, but can collide
                baseCollisionShapeIndex=sphere_collision,
                baseVisualShapeIndex=visual,
                basePosition=pos,
                physicsClientId=self.client_id
            )
            self.sphere_ids.append(sphere_id)

        # Create cylinders with collision
        for i in range(len(points) - 1):
            start = np.array(points[i])
            end = np.array(points[i + 1])
            mid = (start + end) / 2
            vec = end - start
            length = np.linalg.norm(vec)

            if length > 1e-6:
                z_axis = np.array([0, 0, 1])
                axis = np.cross(z_axis, vec)
                axis_norm = np.linalg.norm(axis)
                if axis_norm < 1e-6:
                    quat = [0, 0, 0, 1]
                else:
                    axis = axis / axis_norm
                    angle = np.arccos(np.dot(z_axis, vec / length))
                    quat = p.getQuaternionFromAxisAngle(axis, angle, physicsClientId=self.client_id)
            else:
                quat = [0, 0, 0, 1]

            cyl_collision = p.createCollisionShape(
                shapeType=p.GEOM_CYLINDER,
                radius=self.cylinder_radius,
                height=length,
                physicsClientId=self.client_id
            )

            cyl_visual = p.createVisualShape(
                shapeType=p.GEOM_CYLINDER,
                radius=self.cylinder_radius,
                length=length,
                rgbaColor=[0.75, 0.75, 0.75, 1.0],
                physicsClientId=self.client_id
            )

            cyl_id = p.createMultiBody(
                baseMass=0.0,  # static
                baseCollisionShapeIndex=cyl_collision,
                baseVisualShapeIndex=cyl_visual,
                basePosition=mid,
                baseOrientation=quat,
                physicsClientId=self.client_id
            )

            self.cylinder_ids.append(cyl_id)
    
    def update_robot_skeleton(self, joint_positions):
        """
        Update the positions of the existing robot skeleton.
        """
        #print(f"{joint_positions.shape=}")
        points = np.vstack((np.array([[0, 0, 0]]), joint_positions[:, :3]))
        #print(f"{points.shape=}")
        # Update spheres
        for i, pos in enumerate(points):
            p.resetBasePositionAndOrientation(
                self.sphere_ids[i],
                pos,
                [0, 0, 0, 1],
                physicsClientId=self.client_id
            )

        # Update cylinders
        for i in range(len(points) - 1):
            start = np.array(points[i])
            end = np.array(points[i + 1])
            mid = (start + end) / 2
            vec = end - start
            length = np.linalg.norm(vec)

            if length > 1e-6:
                z_axis = np.array([0, 0, 1])
                axis = np.cross(z_axis, vec)
                axis_norm = np.linalg.norm(axis)
                if axis_norm < 1e-6:
                    quat = [0, 0, 0, 1]
                else:
                    axis = axis / axis_norm
                    angle = np.arccos(np.dot(z_axis, vec / length))
                    quat = p.getQuaternionFromAxisAngle(axis, angle, physicsClientId=self.client_id)
            else:
                quat = [0, 0, 0, 1]

            # Note: you may need to recreate the cylinder if the length changed.
            p.resetBasePositionAndOrientation(
                self.cylinder_ids[i],
                mid,
                quat,
                physicsClientId=self.client_id
            )



    def step(self, action,bool_base=False,bool_render=False):
        """Apply action, update simulation, calculate reward, and return observation."""

        #Time :: t. Preprocess##########
        #Convert end-effector point into the joint space.
        pos_target = []
        orient_target=[]
        #if self.bool_base:#baseline model -> target position is the synchronized motion.
        #    #get the current observation
        #    obs = self._get_observation()
        #    action = self._update_target_sync(obs=obs,action=action)
        for i, p_target in enumerate(action):
            if isinstance(p_target, (list, np.ndarray)):  # Check if vel is a list or array
                p_target = p_target[0]  # Take the first element if it is
            
            if i<3:
                pos_target.append(p_target)#(x,y,z)
            else:
                orient_target.append(p_target) #rotation vector.
        
        #convert quaternion to rotation vector.
        rotvec_target = orient_target#self.quaternion_to_rotvec(q=orient_target)

        #get the current robot's configuration.
        #get the current observation
        obs = self._get_observation()
        p_eef_current = obs[:3] #x,y,z
        quat_eef_current = obs[3:6] #rotation vector #q.x,q.y,q.z,q.w
        rotvec_current = quat_eef_current.copy() #self.quaternion_to_rotvec(q=quat_eef_current)
        #print(f"p_eef={p_eef_current}, quaternion_eef={quat_eef_current}, rotVec={rotvec_current}, p_target = {pos_target}, quat_target={orient_target}, rotVec_target={rotvec_target}")
        
        #calculate the robot's end-effector velocity.
        vel_eef = self.pos2vel(pos_target,rotvec_target,p_eef_current,rotvec_current)
        self.v_eef = vel_eef.copy()
        
        #print(f"joint_angles={joint_angles}")
        #2. calculate the whole robot's configuration.
        self._ur_ctrl.cal_pose_all(self.joints_list)
        #3. calculate jacobian -> self._ur_ctrl.J
        self._ur_ctrl.cal_jacobian(self.joints_list)
        #4. calculate the inverse jacobian.
        invJ = self._ur_driver._inv_jacobian(self._ur_ctrl.J)
        #5. convert eef velocity to joint angular velocity.
        q_dot = invJ @ vel_eef
        self.omega_joint = q_dot.copy()

        ###### Execute ACTION ###########################
        #update robot's joint angles
        for i, velocity in enumerate(q_dot):
            self.joints_list[i] += velocity*self.dt_sample
        
        self.joint_storage.append(self.joints_list.tolist())
        
        #update robot's configuration.
        self._ur_ctrl.cal_pose_all(self.joints_list)
        #update end-effector pose.
        self.pose_eef = self._ur_ctrl.pose6
        #update robot's configuration.
        self.config_robot[0,:]=self._ur_ctrl.pose1
        self.config_robot[1,:]=self._ur_ctrl.pose2
        self.config_robot[2,:]=self._ur_ctrl.pose3
        self.config_robot[3,:]=self._ur_ctrl.pose4
        self.config_robot[4,:]=self._ur_ctrl.pose5
        self.config_robot[5,:]=self._ur_ctrl.pose6
        
        #update robot-side rope's position.
        self.update_target_position_root()
        #update robot's rendering.
        self.update_robot_skeleton(joint_positions=self.config_robot)
        
        # Update human's wrist position.
        dt = 1.0/self.frequency  # assuming 240 Hz simulation
        self.update_target_position(dt)

        # Step simulation
        p.stepSimulation(physicsClientId=self.client_id)
        self.time_step += 1
        ### done.

        ##### Time: t+1 :: POSTPROCESS ###############################
        ## Get midpoint of the rope
        self.rope_mid_point = self.calculate_rope_midpoint()
        
        #get the current robot's configuration.
        #get the current observation
        obs = self._get_observation()

        # Calculate reward
        #imitation-based
        #reward = self._calculate_reward_imitate(obs)
        #general reward.
        reward = self._calculate_long_term(obs)

        #calculate the rope's middle point
        p_rope_middle = self.calculate_rope_midpoint()
        vel_rope_middle = np.zeros(3)
        if (len(self.middle_rope_list)>=2):#more than one step data is available.
            vel_rope_middle = (np.array(self.middle_rope_list[-1])-np.array(self.middle_rope_list[-2]))/self.dt_sample#v(t)-v(t-\Delta t)
        state_rope_middle = np.hstack((np.array([self.rope_length]), p_rope_middle, vel_rope_middle)) #(7,)
        #print(f"state_rope_middle.shape={state_rope_middle.shape}")

        #Return normalized observation.
        #Normalize observation with Min-Max method.

        #normalize only the norm. Adopt the sign separately.
        obs_normalized = obs.copy() #copy the observation.
        #position.
        obs_normalized[:3] = np.array([np.sign(e)*(abs(e)-p_min)/(p_max-p_min) for e,p_min,p_max in zip(obs[:3],self._p_min,self._p_max)])
        #rotation vector.
        obs_normalized[3:6] = np.array([np.sign(q)*(abs(q)-0.0)/(self._q_max-0.0) for q in obs[3:6]])
        #end-effector linear velocity. (vx,vy,vz,wx,wy,wz)
        obs_normalized[6:9] = np.array([np.sign(v)*(abs(v)-0.0)/(self._v_max-0.0) for v in obs[6:9]])
        #end-effector angular velocity.
        obs_normalized[9:12] = np.array([np.sign(w)*(abs(w)-0.0)/(self._omega_max-0.0) for w in obs[9:12]])
        #human's wrist position
        obs_normalized[12:15] = np.array([np.sign(e)*(abs(e)-p_min)/(p_max-p_min) for e,p_min,p_max in zip(obs[12:15],self._p_min,self._p_max)])
        #human's linear velocity. (vx,vy,vz)
        obs_normalized[15:18] = np.array([np.sign(v)*(abs(v)-0.0)/(self._v_max-0.0) for v in obs[15:18]])
        #force
        obs_normalized[18:21] = np.array([np.sign(f)*(abs(f)-self._force_min)/(self._force_max-self._force_min) for f in obs[18:21]])
        #torque
        obs_normalized[21:24] = np.array([np.sign(t)*(abs(t)-self._torque_min)/(self._torque_max-self._torque_min) for t in obs[21:24]])
        #rope state
        #rope length
        obs_normalized[24] = obs[24]/self._l_rope_max
        #rope middle position.
        obs_normalized[25:28] = np.array([np.sign(p)*(abs(p)-p_min)/(p_max-p_min) for p,p_min,p_max in zip(obs[25:28],self._p_min,self._p_max)])
        #rope middle velocity.
        obs_normalized[28:31] = np.array([np.sign(v)*(abs(v)-0.0)/(self._v_max-0.0) for v in obs[28:31]])

        # Check termination conditions
        done = False
        if self.time_step >= self.max_episode_steps:
            done = True
        
        return obs_normalized,state_rope_middle, reward, done, {}
    
    def _get_observation(self):
        """Retrieve the current state of the environment."""
        #human's current position.
        #self.pos_human_current : current human position [x,y,z]
        #self.vel_human : current human's velocity [vx,vy,vz]

        end_effector_position = self.pose_eef[:3]  # Get the position (x, y, z)
        rotVec = self.pose_eef[3:]  # Get the orientation (quaternion)
        end_effector_orientation = rotVec.copy() #get the rotatino vector. #self.rotation_vector_to_quaternion(rot_vec=rotVec)
        #robot's end-effector velocity

        # # Get joint states
        joint_angles = self.joints_list.copy()
        joint_velocities = self.omega_joint.copy() #joints' angular velocity.
        #calculate the inverse jacobian
        #2. calculate the whole robot's configuration.
        self._ur_ctrl.cal_pose_all(joint_angles)
        #3. calculate jacobian -> self._ur_ctrl.J
        self._ur_ctrl.cal_jacobian(joint_angles)
        #4. calculate the end-effector velocity (6,).[vx,vy,vz,omega_x,omega_y,omega_z]
        vel_eef = self._ur_ctrl.J@joint_velocities  
        #vel_eef = np.array([vel_eef[0],vel_eef[1],vel_eef[2],vel_eef[3],vel_eef[4],vel_eef[5]]) #vx,vy,vz,adjust the velocity to match the eef pose.


        """Contact force estimation"""
        # Estimate the end-effector force/torque from the multi-link rope
        net_force = [0, 0, 0]
        net_torque = [0, 0, 0]
        # Get end-effector position
        eef_pos = self.pose_eef[:3]
        
        # Calculate forces and torques from all rope links
        for i, link_id in enumerate(self.rope_link_ids):
            # Get link state
            link_state = p.getBasePositionAndOrientation(link_id, physicsClientId=self.client_id)
            link_pos = link_state[0]
            link_vel, link_angular_vel = p.getBaseVelocity(link_id, physicsClientId=self.client_id)
            
            # Calculate distance from end-effector to this link
            r_vec = [lp - ep for lp, ep in zip(link_pos, eef_pos)]
            r_magnitude = np.linalg.norm(r_vec)
            
            # Skip if link is too far from end-effector (not directly connected)
            if r_magnitude > self.link_length *500:  #consider 50 cm from the end-effector. # Allow for some tolerance
                continue
            
            # Calculate force from this link
            # Force due to gravity on the link
            gravity_force = [0, 0, -9.81 * self.link_mass]
            
            # Force due to tension (if this is the first link, it's directly connected)
            tension_force = [0, 0, 0]
            if i == 0:  # First link is directly attached to end-effector
                # Calculate tension based on link velocity and acceleration
                if hasattr(self, 'prev_link_vels') and i < len(self.prev_link_vels):
                    dt = 1.0/self.frequency
                    acc = [(v - vp) / dt for v, vp in zip(link_vel, self.prev_link_vels[i])]
                    tension_force = [self.link_mass * a for a in acc]
                else:
                    # Initialize previous velocities if not available
                    if not hasattr(self, 'prev_link_vels'):
                        self.prev_link_vels = {}
                    self.prev_link_vels[i] = link_vel
                    tension_force = [0, 0, 0]
            
            # Total force from this link
            link_force = [gf + tf for gf, tf in zip(gravity_force, tension_force)]
            
            # Add to net force
            net_force = [nf + lf for nf, lf in zip(net_force, link_force)]
            
            # Calculate torque = r × F
            if r_magnitude > 0.001:  # Avoid division by zero
                link_torque = [
                    r_vec[1] * link_force[2] - r_vec[2] * link_force[1],
                    r_vec[2] * link_force[0] - r_vec[0] * link_force[2],
                    r_vec[0] * link_force[1] - r_vec[1] * link_force[0]
                ]
                net_torque = [nt + lt for nt, lt in zip(net_torque, link_torque)]
            
            # Update previous velocities for next iteration
            if not hasattr(self, 'prev_link_vels'):
                self.prev_link_vels = {}
            self.prev_link_vels[i] = link_vel
        
        # Add contact forces from rope links touching the end-effector
        # Check for contact between rope links and end-effector
        contact_force = [0, 0, 0]
        contact_torque = [0, 0, 0]
        
        # Get contact points between the rope root link and the robot end-effector
        # self.root_link_id is the rope link attached to the end-effector
        # self.point_id_root is the robot's end-effector
        if hasattr(self, 'root_link_id') and hasattr(self, 'point_id_root'):
            contact_points = p.getContactPoints(
                bodyA=self.root_link_id,  # Rope root link
                bodyB=self.point_id_root,  # Robot end-effector
                physicsClientId=self.client_id
            )
            
            for contact in contact_points:
                # Contact force in world coordinates
                contact_normal = contact[7]  # Contact normal
                contact_force_magnitude = contact[9]  # Contact force magnitude
                
                # Convert to force vector
                contact_force_vec = [cn * contact_force_magnitude for cn in contact_normal]
                contact_force = [cf + ccf for cf, ccf in zip(contact_force, contact_force_vec)]
                
                # Contact point relative to end-effector
                contact_point = contact[5]  # Contact point on body B (end-effector)
                r_contact = [cp - ep for cp, ep in zip(contact_point, eef_pos)]
                
                # Torque from contact force
                contact_torque_vec = [
                    r_contact[1] * contact_force_vec[2] - r_contact[2] * contact_force_vec[1],
                    r_contact[2] * contact_force_vec[0] - r_contact[0] * contact_force_vec[2],
                    r_contact[0] * contact_force_vec[1] - r_contact[1] * contact_force_vec[0]
                ]
                contact_torque = [ct + cct for ct, cct in zip(contact_torque, contact_torque_vec)]
        
        # Add contact forces and torques to net values
        net_force = [nf + cf for nf, cf in zip(net_force, contact_force)]
        net_torque = [nt + ct for nt, ct in zip(net_torque, contact_torque)]
        
        # Apply low-pass filter to smooth the force/torque estimates
        if not hasattr(self, 'filtered_force'):
            self.filtered_force = [0, 0, 0]
            self.filtered_torque = [0, 0, 0]
        
        alpha = 0.1  # Filter coefficient (0 = no filtering, 1 = no change)
        self.filtered_force = [ff * (1-alpha) + nf * alpha for ff, nf in zip(self.filtered_force, net_force)]
        self.filtered_torque = [ft * (1-alpha) + nt * alpha for ft, nt in zip(self.filtered_torque, net_torque)]
        
        # Use filtered values
        net_force = self.filtered_force
        net_torque = self.filtered_torque
        """END OF Contact force estimation"""

        #get the rope's state.
        p_rope_middle = self.calculate_rope_midpoint()
        vel_rope_middle = np.zeros(3)
        if (len(self.middle_rope_list)>=2):#more than one step data is available.
            vel_rope_middle = (np.array(self.middle_rope_list[-1])-np.array(self.middle_rope_list[-2]))/self.dt_sample#v(t)-v(t-\Delta t)
        state_rope_middle = np.hstack((np.array([self.rope_length]),p_rope_middle, vel_rope_middle)) #(7,)
        
        # Observation: [Joint Angles, Joint Velocities, Target Position, Rope Midpoint]
        # Observation: [Joint Angles, Joint Velocities, Target Position, Rope Midpoint, End Effector Position, End Effector Orientation]

        return np.concatenate([
            end_effector_position, #3
            end_effector_orientation, #3 (6)
            vel_eef, #6 (12)
            self.target_position, #3 (15)
            self.vel_human, #3 (18)
            net_force, #3 (21)
            net_torque, #3 (24)
            state_rope_middle  #7. (length,pos,vel) (31)
        ]).astype(np.float32)
    
    def calculate_rope_midpoint(self):
        """Calculate the midpoint of the rope based on the positions of the first and last links."""
        p1, _ = p.getBasePositionAndOrientation(self.rope_link_ids[int(self.num_links//2)-1], physicsClientId=self.client_id)
        p2, _ = p.getBasePositionAndOrientation(self.rope_link_ids[int(self.num_links//2)], physicsClientId=self.client_id)
        midpoint = [(f + l)/2 for f, l in zip(p1,p2)]
        return np.array(midpoint)
    
    def _update_target_sync(self,obs,action):
        """Compute the reward based on imitating human motion.
        Parameters:
        -------------
        obs (numpy.ndarray) :               
            end_effector_position, #3 (-31)
            end_effector_orientation, #3 (6) (-28)
            vel_eef, #6 (12) (-25)
            self.pos_human_current, #3 (15) (-19)
            self.vel_human, #3 (18) (-16)
            net_force, #3 (21) (-13)
            net_torque, #3 (24) (-10)
            state_rope_middle  #7. (length,pos,vel) (31)
        
        Returns:
        -------------
        action (np.ndarray) : action_return (x,y,z,rot_vec)
        """

        end_effector = obs[:3].copy()#end-effector distance
        action_return = action.copy()
        #save target positions.
        if (self.target_robot[-1][-1]==1e5):#still not available
            if (self.target_robot[0][0]==1e5):#first element is still 0.
                self.target_robot[0] = np.array(self.target_position) #Human's wrist position.
            elif (self.target_robot[1][0]==1e5):#second element is still 0.
                self.target_robot[1] = np.array(self.target_position)
            elif (self.target_robot[2][0]==1e5):#third element is still 0.
                self.target_robot[2] = np.array(self.target_position)
        else:#3 data is gathered.
            
            #update data
            idx_save = self.counter_save_robot%3 #new data
            self.target_robot = self.target_robot[1:, :].copy() #remove the first (oldest) element.
            self.target_robot = np.vstack((self.target_robot, self.target_position))

            ## POSITION ##
            #calculate the normal vector from the 3 taregt positions.
            normal_vector = self.calculate_perpendicular_vector(self.target_robot[0],self.target_robot[1],self.target_robot[2])
            #print(normal_vector)

            #calculate the robot's end-effector ideal position
            ideal_robot_ee_pos = np.array(self.target_position)+normal_vector*self.rope_length/(2**(0.5)) #positive candidate
            ideal_robot_ee_neg = np.array(self.target_position)-normal_vector*self.rope_length/(2**(0.5)) #negative candidate

            #calculate the distance between the ideal position and current position
            dist_ee2ideal_pos = np.linalg.norm(end_effector-ideal_robot_ee_pos) #ee to positive ideal.
            dist_ee2ideal_neg = np.linalg.norm(end_effector-ideal_robot_ee_neg) #ee to negative ideal.

            dist_ee2ideal = min(dist_ee2ideal_pos,dist_ee2ideal_neg) #distance [m]. absolute value.
            bool_positive = True
            if abs(dist_ee2ideal-dist_ee2ideal_neg)<=1.0e-3:
                bool_positive = False

            if bool_positive:
                #position.
                action_return[:3] = ideal_robot_ee_pos.copy() 
                #rotation vector.
                normal_vector_eval = np.array([-v for v in normal_vector]) #reverse
                action_return[3:] = normal_vector_eval.copy()
            else:
                #position.
                action_return[:3] = ideal_robot_ee_neg.copy()
                #rotation vector
                action_return[3:] = normal_vector.copy()

        return action_return

    def _calculate_reward_imitate(self,obs):
        """Compute the reward based on imitating human motion.
        Parameters:
        -------------
        obs (numpy.ndarray) :               
            end_effector_position, #3 (-31)
            end_effector_orientation, #3 (6) (-28)
            vel_eef, #6 (12) (-25)
            self.pos_human_current, #3 (15) (-19)
            self.vel_human, #3 (18) (-16)
            net_force, #3 (21) (-13)
            net_torque, #3 (24) (-10)
            state_rope_middle  #7. (length,pos,vel) (31)
        
        Returns:
        -------------
        reward (float) : scalar reward.
        """

        end_effector = obs[:3].copy()#end-effector distance

        #save target positions.
        if (self.target_robot[-1][-1]==1e5):#still not available
            if (self.target_robot[0][0]==1e5):#first element is still 0.
                self.target_robot[0] = np.array(self.target_position) #Human's wrist position.
            elif (self.target_robot[1][0]==1e5):#second element is still 0.
                self.target_robot[1] = np.array(self.target_position)
            elif (self.target_robot[2][0]==1e5):#third element is still 0.
                self.target_robot[2] = np.array(self.target_position)
            reward = 0
        else:#3 data is gathered.
            
            #update data
            idx_save = self.counter_save_robot%3 #new data
            self.target_robot = self.target_robot[1:, :].copy() #remove the first (oldest) element.
            self.target_robot = np.vstack((self.target_robot, self.target_position))

            ## POSITION ##
            #calculate the normal vector from the 3 taregt positions.
            normal_vector = self.calculate_perpendicular_vector(self.target_robot[0],self.target_robot[1],self.target_robot[2])
            #print(normal_vector)

            #calculate the robot's end-effector ideal position
            ideal_robot_ee_pos = np.array(self.target_position)+normal_vector*self.rope_length/(2**(0.5)) #positive candidate
            ideal_robot_ee_neg = np.array(self.target_position)-normal_vector*self.rope_length/(2**(0.5)) #negative candidate

            #calculate the distance between the ideal position and current position
            dist_ee2ideal_pos = np.linalg.norm(end_effector-ideal_robot_ee_pos) #ee to positive ideal.
            dist_ee2ideal_neg = np.linalg.norm(end_effector-ideal_robot_ee_neg) #ee to negative ideal.

            dist_ee2ideal = min(dist_ee2ideal_pos,dist_ee2ideal_neg) #distance [m]. absolute value.
            bool_positive = True
            if abs(dist_ee2ideal-dist_ee2ideal_neg)<=1.0e-3:
                bool_positive = False

            ### FINISH POSITION ###
        
            ### ORIENTATION ###
            #current robot's end-effector orientation in rotation vector. calculate the current robot pose.
            orientation_robot = obs[3:6]
            #ideal pose
            if bool_positive:#positive normal vector -> ideal pose is -1.0*normal_vector
                normal_vector_eval = np.array([-v for v in normal_vector]) #reverse
                cos_similarity = self.evaluate_alignment(orientation_robot,normal_vector_eval) #[rad]

            else: #negative normal vector -> ideal pose is normal_vector.
                cos_similarity = self.evaluate_alignment(orientation_robot,normal_vector) #[rad]
            #cos_similarity = np.dot(v1,v2) : the more similar, closer to 1. 
            #radian = abs(radian) #positive

            ## FINISH ORIENTATION ##

            ## TOTAL REWARD ##
            if not np.isnan(dist_ee2ideal * cos_similarity):
                reward = -1.0*dist_ee2ideal*(2-cos_similarity)#-1*err*(2-cos) #larger error, or smaller cos_similarity is panished. -(dist_ee2ideal * (1+cos_similarity))#/self.rope_length #[m]*[rad] = [m]. peripheral length of the circle.
            else:
                reward = 0.0 

        distance_robot2target = np.linalg.norm(end_effector-self.target_position)#(end_effector[0]-self.target_position[0])**2+(end_effector[1]-self.target_position[1])**2+(end_effector[2]-self.target_position[2])**2
        #distance_robot2target = distance_robot2target**(0.5)#length
        if distance_robot2target>=self.rope_length: #length is longer than the rope lengtj
            reward =reward + self._penalty_length#*= 2
        
        return reward

    def _calculate_long_term(self, obs):
        """Calculate the long-term reward:
        Parameters:
        -------------
        obs (numpy.ndarray) :               
            end_effector_position, #3 (-31)
            end_effector_orientation, #3 (6) (-28)
            vel_eef, #6 (12) (-25)
            self.pos_human_current, #3 (15) (-19)
            self.vel_human, #3 (18) (-16)
            net_force, #3 (21) (-13)
            net_torque, #3 (24) (-10)
            state_rope_middle  #7. (length,pos,vel) (31)
        
        Returns:
        -------------
        reward (float) : scalar reward.
        """
        #robot, human and turning samples configuration.
        #prepare figures
        if self._bool_debug:
            fig3,ax3 = plt.subplots(2,2,figsize=(16,16),subplot_kw={'projection': '3d'})
            for j in range(4):
                # Customize the plot
                ax3[j//2,j%2].set_xlabel('X [m]')
                ax3[j//2,j%2].set_ylabel('Y [m]')
                ax3[j//2,j%2].set_zlabel('Z [m]')
                ax3[j//2,j%2].set_xlim(-1.5,1.0)
                ax3[j//2,j%2].set_ylim(-1.0,1.5)
                ax3[j//2,j%2].set_zlim(0,1)
            ax3[0,1].view_init(elev=0, azim=90) #from x
            ax3[1,0].view_init(elev=0, azim=0)#from Y
            ax3[1,1].view_init(elev=90, azim=-90)#from z
        
        #get the middle point : [x,y,z]
        p_rope_middle = self.calculate_rope_midpoint()
        #get human position
        pos_human = obs[12:15]
        pos_eef = obs[:3]

        # Spatial reward ##
        #Reward 1. Maximize the amplitude.
        #> pos_human (obs[14:17]), pos_eef(obs[:3]), p_rope_middle, self.p_rope_middle, self.middle_rope_list = [] #storage for saving middle points
        #1. vec_h2r=p_r-p_h, p_mid=(p_h+p_r)/2
        vec_h2r = pos_eef-pos_human
        p_mid = (pos_eef+pos_human)/2.0
        #2. if (len(self.middle_rope_list)>=3) p_mid2i = p_i - p_mid
        self.middle_rope_list.append(p_rope_middle)
        reward_amplitude = 0
        if (len(self.middle_rope_list)>=self._max_size):
            self.middle_rope_list = self.middle_rope_list[1:]#remove the front element.
        if (len(self.middle_rope_list)>=3):#more than 3 points. => can fit with circle.
            p_project_list = []
            for i in range(1,4):#latest 3 points are adopted.
                p_i = self.middle_rope_list[-i]
                p_mid2i = p_i - p_mid
                #3. p_mid2h = np.dot(p_mid2i,vec_h2r)/(np.linalg.norm(vec_h2r)**2)*vec_h2r
                p_mid2h = np.dot(p_mid2i,vec_h2r)/(np.linalg.norm(vec_h2r)**2)*vec_h2r
                #4. p_project = vec_h2i = p_mid2i-p_mid2h
                p_project = p_mid2i-p_mid2h #origin should be the middle point of the rope.
                #save: #5. [p1_project,p2_project, p2_project]
                p_project_list.append(p_project)
        
            #6. calculate the perpendicular vecotr to vec_h2r. -> temp -> u1 -> u2. p_prject=a*u1+b*u2. (a,b) -> planar circle fitting. (x-x0)**2+(y-y0)**2=r
            #calculate the unit two perpendicular vector.
            u1,u2 = self.compute_orthogonal_vectors(v=vec_h2r)
            # Stack them as columns to form a 3x2 matrix A
            A = np.column_stack((u1, u2))  # shape: (3, 2)
            #print(f"A.shape={A.shape}, dot_u1_h2r={round(np.dot(u1,vec_h2r),2)},dot_u2_h2r={round(np.dot(u2,vec_h2r),2)}, dot_u1_u2={round(np.dot(u1,u2),2)}")

            # Convert list to numpy array for easier matrix operations
            p_project_array = np.array(p_project_list)  # shape: (3, 3) if 3 points

            # Solve for (x, y) coordinates of each projected point in the plane
            xy_list = [] #(3,2)
            for p_proj in p_project_array:
                x_y = A.T @ p_proj #(2,) A is orthogonal matrix, so we can skip A.T@A. np.linalg.inv(A.T @ A) @ A.T @ p_proj #
                xy_list.append(x_y.tolist())
            
            x_center,y_center,radius = self.fit_circle_to_3_points(xy_list=xy_list)

            #7. p_center = x0*u1+y0*u2. deviate = |p_center-p_mid|
            p_center =p_mid+x_center*u1+y_center*u2 # 3D position.
            z_min = p_center[2]-radius #the bottom point. This should be around 0 ideally.
            deviation = abs(z_min) #distance from the ground surface.
            #deviation = np.linalg.norm(p_center-p_mid)

            #8. reward = (r - deviate) / self.rope_length
            reward_amplitude = radius/self.rope_length#(radius-deviation)/self.rope_length

            #update radius for the temporal reward
            self.r_current = radius
        
            #plot spatial reward 
            if self._bool_debug:
                for i in range(4):
                    #two planar vector.
                    # Draw a 3D arrow using quiver
                    ax3[i//2,i%2].plot(p_center[0], p_center[1], p_center[2], 
                            color='k', marker="o",markersize=2,label="center")
                    ax3[i//2,i%2].quiver(p_center[0], p_center[1], p_center[2], 
                            u1[0], u1[1], u1[2], 
                            color='k', arrow_length_ratio=0.2, linewidth=2,label="unit vector")
                    ax3[i//2,i%2].quiver(p_center[0], p_center[1], p_center[2], 
                            u2[0], u2[1], u2[2], 
                            color='k', arrow_length_ratio=0.2, linewidth=2)
                    for j in range(len(p_project_list)):#for each project point.
                        p_=p_project_list[j]
                        ax3[i//2,i%2].plot(p_[0], p_[1], p_[2], color='k', marker="*",markersize=1)#,label="center")
                    for j in range(len(self.middle_rope_list)):#for each 3d point.
                        p_=self.middle_rope_list[j]
                        ax3[i//2,i%2].plot(p_[0], p_[1], p_[2], color='k', marker="x",markersize=1)#,label="center")
                
                print(rf"Reward amplitude :: radius={round(radius,2)},zmin={round(z_min,2)},reward_amp={round(reward_amplitude,2)}")
                
        #Reward 2. Convexity. number of crossing point.
        #> pos_human (obs[14:17]), pos_eef(obs[:3]), p_rope_middle, rope's point ()
        #1. prepare the baseline vector. vec_r2mid, vec_h2mid.
        #2. for each point by middle point. for i in range(int(self.num_links//2))
        signs_r2mid = [] #+1 or -1
        idx_r2mid = []
        signs_h2mid = []
        idx_h2mid = []
        rope_point_list = []
        #print(f"type(self.rope_link_ids)={type(self.rope_link_ids)}, len(self.rope_link_ids)={len(self.rope_link_ids)}")
        for i in range(len(self.rope_link_ids)):#for each mass
            body_id = self.rope_link_ids[i]
            #print(f"Invalid body ID at index {i}: {body_id}")
            #position and orientation.
            p_i, _ = p.getBasePositionAndOrientation(body_id, physicsClientId=self.client_id)#i-th mass
            rope_point_list.append(p_i)
            #compare the value of cosine. 
            if (4<=i and i<len(self.rope_link_ids)-4) and i<(self.num_links/2):#robot2middle
                #cos_r2i= np.dot(vec_r2i,vec_r2h)/((np.linalg.norm(vec_r2i)+1e-10)*(np.linalg.norm(vec_r2h)+1e-10))
                vec_r2i = p_i-pos_eef
                vec_r2h = pos_human-pos_eef 
                cos_r2i= np.dot(vec_r2i,vec_r2h)/((np.linalg.norm(vec_r2i)+1e-10)*(np.linalg.norm(vec_r2h)+1e-10))
                #cos_r2mid= np.dot(vec_r2mid,vec_r2h)/((np.linalg.norm(vec_r2mid)+1e-10)*(np.linalg.norm(vec_r2h)+1e-10))
                vec_r2mid = p_rope_middle - pos_eef
                cos_r2mid= np.dot(vec_r2mid,vec_r2h)/((np.linalg.norm(vec_r2mid)+1e-10)*(np.linalg.norm(vec_r2h)+1e-10))
                #If cos_r2i<=cos_r2mid: positive. else: negative.
                sign_p = +1
                if (cos_r2i>cos_r2mid):#concave.
                    sign_p = -1
                #make a binary list including positive and negative.[++...--..+-]
                signs_r2mid.append(sign_p)
                idx_r2mid.append(i)
            elif (4<=i and i< len(self.rope_link_ids)) and i>(self.num_links/2): #human2middle
                #cos_r2i= np.dot(vec_r2i,vec_r2h)/((np.linalg.norm(vec_r2i)+1e-10)*(np.linalg.norm(vec_r2h)+1e-10))
                vec_h2i = p_i-pos_human
                vec_h2r = pos_eef-pos_human 
                cos_h2i= np.dot(vec_h2i,vec_h2r)/((np.linalg.norm(vec_h2i)+1e-10)*(np.linalg.norm(vec_h2r)+1e-10))
                #cos_r2mid= np.dot(vec_r2mid,vec_r2h)/((np.linalg.norm(vec_r2mid)+1e-10)*(np.linalg.norm(vec_r2h)+1e-10))
                vec_h2mid = p_rope_middle - pos_human
                cos_h2mid= np.dot(vec_h2mid,vec_h2r)/((np.linalg.norm(vec_h2mid)+1e-10)*(np.linalg.norm(vec_h2r)+1e-10))
                #If cos_r2i<=cos_r2mid: positive. else: negative.
                sign_p = +1
                if (cos_h2i>cos_h2mid):#concave.
                    sign_p = -1
                #make a binary list including positive and negative.[++...--..+-]
                signs_h2mid.append(sign_p)
                idx_h2mid.append(i)

        #calculate the number of sign switching.
        n_sign_switch = 0
        idx_switch = []
        for i in range(1,len(signs_r2mid)-1):#robot2middle
            if signs_r2mid[i-1]*signs_r2mid[i]<=0:#sign has changed.
                n_sign_switch+=1
                idx_switch.append(idx_r2mid[i])
        for i in range(1,len(signs_h2mid)-1):#human2middle
            if signs_h2mid[i-1]*signs_h2mid[i]<=0:#sign has changed.
                n_sign_switch+=1
                idx_switch.append(idx_h2mid[i])
        
        #reward for convexity. Todo : have to normalize considering other elements.
        reward_convex = -1*min(n_sign_switch*self._unit_length,self._w_penalty_convex*abs(reward_amplitude))

        #plot rope condition.
        if self._bool_debug:
            for i in range(4):
                for j in range(len(rope_point_list)):#for each project point.
                    p_=rope_point_list[j]
                    color="b"
                    marker="o"
                    if j in idx_switch:
                        color="r"
                        marker="x"
                    if j==0:
                        ax3[i//2,i%2].plot(p_[0], p_[1], p_[2], color=color, marker=marker,markersize=1,label="rope")
                    else:
                        ax3[i//2,i%2].plot(p_[0], p_[1], p_[2], color=color, marker=marker,markersize=1)#,label="center")
            print(rf"Reward for convexity :: n_switch={n_sign_switch},reward={round(reward_convex,2)}")
        
        #Temporal reward ##
        #Reward 3. continuous turning radius. smaller better.
        #> self.r_current, self.r_previous
        reward_radius_change = 0.0
        if self.r_current>0.0 and self.r_previous>0.0:
            reward_radius_change = (self.r_current-self.r_previous)/self.rope_length
        
        if self._bool_debug:
            print(f"Reward change in radius :: r_curr={round(self.r_current,2)},r_prev={round(self.r_previous,2)}"+rf"reward_dr={round(reward_amplitude,2)}")
        
        #save current radius as a previous data for the next step.
        self.r_previous = self.r_current 

        #Reward 4. Continuous rope's velocity. Predictability of the middle point.        
        #> predictability. self.center_prev:np.array([x,y,z]), self.r_previous
        reward_prediction=0.0
        if (len(self.middle_rope_list)>=4):#sufficient number of points.
            #predict with previous 3D sphere fitting.
            #step1. calculate the difference from the previous fitting parameters.
            xc=self.center_prev[0]
            yc=self.center_prev[1]
            zc=self.center_prev[2]
            r =self.center_prev[3]
            if (r>0.0):
                reward_prediction = -1.0*(abs((p_rope_middle[0]-xc)**2+(p_rope_middle[1]-yc)**2+(p_rope_middle[2]-zc)**2-r**2))**(0.5)/self.rope_length#squared root difference.
            
            #step2. update fitting parameters. self.center_prev:np.array([x,y,z,r]), self.r_previous
            #calculate from 3 points. 4-> fitting. 5~-> RLS-method.
            points_fit = np.array(self.middle_rope_list[-4:])#extract the latest 4 points. (4,3)
            x0, y0, z0, r0 = self.fit_sphere_to_points(points=points_fit) #calculate the parameters for fitting sphere.
            self.center_prev[0]=x0
            self.center_prev[1]=y0
            self.center_prev[2]=z0
            self.center_prev[3]=r0

            #plot a sphere

            # Create sphere data
            u = np.linspace(0, 2 * np.pi, 30)
            v = np.linspace(0, np.pi, 30)
            x = z0 + r0 * np.outer(np.cos(u), np.sin(v))
            y = y0 + r0 * np.outer(np.sin(u), np.sin(v))
            z = z0 + r0 * np.outer(np.ones(np.size(u)), np.cos(v))

            # Plot the surface
            #for i in range(4):
            #    ax3[i//2,i%2].plot_surface(x, y, z, color='b', alpha=0.05, edgecolor='b',label=f"reward_sphere={round(reward_prediction,2)}")
            
            if self._bool_debug:
                print(f"Reward sphere (predictabiity) :: reward_sphere={round(reward_prediction,2)}")
        #Reward 5. Force and torque penalty.
        #> pos_human (obs[14:17]), pos_eef(obs[:3]), force/torque (obs[-6:])
        #1 direction from robot to the human, vec_r2h. 
        #calculate the force along the direction toward the human. -> penalty.
        vec_r2h = pos_human-pos_eef
        force = obs[-13:-10]
        torque = obs[-10:]
        force_r2h = abs(np.dot(force,vec_r2h)/((np.linalg.norm(force)+1e-10)*(np.linalg.norm(vec_r2h)+1e-10)))

        #plot spatial reward 
        if self._bool_debug:
            for i in range(4):
                #two planar vector.
                # Draw a 3D arrow using quiver
                ax3[i//2,i%2].plot(pos_human[0],pos_human[1],pos_human[2],#p_center[0], p_center[1], p_center[2], 
                        color='orange', marker="o",markersize=3,label="human")
                ax3[i//2,i%2].plot(pos_eef[0],pos_eef[1],pos_eef[2],#p_center[0], p_center[1], p_center[2], 
                        color='lightblue', marker="o",markersize=3,label="robot")
                ax3[i//2,i%2].quiver(pos_eef[0],pos_eef[1],pos_eef[2], 
                        force[0], force[1], force[2], 
                        color='green', arrow_length_ratio=0.2, linewidth=2,label=r"$force_{robot}$")
            print(rf"Reward force :: $cos_r2h$={round(force_r2h,2)}")

        #synchronous force to the robot's end-effector.
        force_sync = 0.0
        if (len(self.middle_rope_list)>=2):#more than one step data is available.
            vel_rope_middle = np.array(self.middle_rope_list[-1])-np.array(self.middle_rope_list[-2])#v(t)-v(t-\Delta t)
            force_sync = -1.0*np.dot(vel_rope_middle,force)/((np.linalg.norm(vel_rope_middle)+1e-10)*(np.linalg.norm(force)+1e-10))

            if self._bool_debug:
                for i in range(4):
                    #two planar vector.
                    ax3[i//2,i%2].quiver(p_rope_middle[0],p_rope_middle[1],p_rope_middle[2], 
                            vel_rope_middle[0], vel_rope_middle[1], vel_rope_middle[2], 
                            color='blue', arrow_length_ratio=0.2, linewidth=2,label=r"$V_{rope}$")
            
                print(rf"Reward force :: cos_sync={round(force_sync,2)}")
        
        reward_force = -1.0*force_r2h + force_sync

        if self._bool_debug:        
            print(f"Reward force :: reward_f={round(reward_force,2)}")


        #distance between human and robot.
        #constant distance.
        d_r2h_current = np.linalg.norm(vec_r2h)
        reward_dist_r2h = -1.0*abs(d_r2h_current-self.dist_r2h_current)/self.rope_length
        self.dist_r2h_current = d_r2h_current

        if self._bool_debug:
            print(f"Reward distance :: dist_h2r={round(d_r2h_current,2)}<{round(self.rope_length,2)}")

        #rope's length penalty.
        penalty_length = 0
        if (d_r2h_current>self.rope_length):
            penalty_length = self._penalty_length#-1.0*abs(reward_amplitude)#self._w_penalty_length*max(1.0,min(3.0,abs(reward_amplitude)))

        #velocity reward.
        vel_eef = obs[6:9] #linear velocity.
        speed_eef = np.linalg.norm(vel_eef)
        reward_speed = self.weight_speed*speed_eef

        #simplified reward function.
        reward_total = reward_amplitude + penalty_length + reward_speed
        #normalize the reward.
        reward_total /= self.step_episode
        #complex reward function.
        #reward_total = reward_amplitude + reward_convex + reward_radius_change + reward_prediction + reward_force + reward_dist_r2h + penalty_length

        # Show plot and wait 2 seconds
        if self._bool_debug:
            ax3[0,0].legend(fontsize=8)
            plt.show(block=False)
            plt.pause(10)  # Wait for 2 seconds
            plt.close(fig3)  # Close the figure window

        return reward_total

    def fit_circle_to_3_points(self,xy_list):
        """
        Fit a circle to 3 points (x, y) and return center (x0, y0) and radius r.
        """
        xy = np.array(xy_list)  # shape: (3, 2)
        x = xy[:, 0]
        y = xy[:, 1]

        # Right-hand side: x^2 + y^2
        b = x**2 + y**2

        # Matrix for the linear system
        A = np.column_stack((x, y, np.ones(3)))

        # Solve for [2*x0, 2*y0, c]
        sol = np.linalg.solve(A, b)

        x0 = sol[0] / 2
        y0 = sol[1] / 2
        c = sol[2]

        r = np.sqrt(x0**2 + y0**2 + c)

        return x0, y0, r

    def fit_sphere_to_points(self,points):
        # Ensure input is 4x3
        assert points.shape == (4, 3), "You must provide 4 points with 3 coordinates each."

        A = []
        B = []

        for p in points:
            x, y, z = p
            A.append([x, y, z, 1])
            B.append(-(x**2 + y**2 + z**2))

        A = np.array(A)
        B = np.array(B)

        # Solve linear system A * params = B
        try:
            cond_A = np.linalg.cond(A)
            if cond_A > 1e12:
                return 0,0,0,0
            else:#a is nonsigular
                params = np.linalg.solve(A, B)

                x0 = -0.5 * params[0]
                y0 = -0.5 * params[1]
                z0 = -0.5 * params[2]
                r_squared = x0**2 + y0**2 + z0**2 - params[3]
                
                # Check if r_squared is valid
                if r_squared < 0:
                    return 0,0,0,0
                    
                r = np.sqrt(r_squared)

                return x0, y0, z0, r
        except (np.linalg.LinAlgError, ValueError):
            # If SVD fails or matrix is singular, return default values
            return 0,0,0,0
    
    def calculate_perpendicular_vector(self,p1, p2, p3):
        v1 = p2 - p1
        v2 = p3 - p1
        n_vector = np.cross(v1, v2)  
        norm_vector = np.linalg.norm(n_vector)
        if norm_vector<1.0e-6:
            return np.array([0,0,0])
        else:
            for i in range(n_vector.shape[0]):
                n_vector[i] = n_vector[i]/norm_vector
            return n_vector  #normalized normal vector.
    
    def pos2vel(self, pos_target,rotvec_target,p_eef_current,rotvec_current):
        """Calculate the velocity from the current situation.
        eef : end-effector

        Args:
            pos_target (numpy.ndarray): target eef position (3)
            rotvec_target (numpy.ndarray): target rotation vector (3)
            p_eef_current (numpy.ndarray): current eef position (3)
            rotvec_current (numpy.ndarray): current eef rotation vector (3)
        
        Returns:
            v_eef(numpy.ndarray) : robot's eef velocity. (6)
        """
        #calculate the speed.
        #velocity [vx,vy,vz]
        vel_eef = self.speed_setting(pos_target,p_eef_current,bool_pos=True)
        #angular velocity [omega_x,omega_y,omega_z]
        omega_eef = self.speed_setting(rotvec_target,rotvec_current,bool_pos=False)
        return np.array([vel_eef[0],vel_eef[1],vel_eef[2],omega_eef[0],omega_eef[1],omega_eef[2]])


    def speed_setting(self,p_target,p_cur,bool_pos):
        """calculate the predefined speed limit.

        Args:
            p_target (numpy.ndarray): target eef position/rotation vector (3).
            p_cur (numpy.ndarray): current eef position/rotation vector (3).
            bool_pos (bool): True : position. False : rotation vector. 
        """
        if bool_pos:#positional difference.
            dist = ((p_target[0]-p_cur[0])**2+(p_target[1]-p_cur[1])**2+(p_target[2]-p_cur[2])**2)**(0.5)
            speed = 0.0
            if dist>=self._d_deccelerate:
                speed = self._v_max_eef
            else:#deccelerating zone.
                speed = self._v_max_eef*(1-np.cos(math.pi/2.0*(dist/self._d_deccelerate)))
            #calculate the velocity.
            velocity = np.array([(speed/dist)*(p_target[0]-p_cur[0]),(speed/dist)*(p_target[1]-p_cur[1]),(speed/dist)*(p_target[2]-p_cur[2])])
            return velocity
            
        else: #rotation vector's difference.
            #calculate the rotation angle.
            angle,rotvec_transform = self.calculate_rotation_angle_between_rotvecs(rotation_vector1=p_cur, rotation_vector2=p_target)
            omega = 0.0
            if angle>=self._theta_deccelerate:
                omega = self._omega_max_eef
            else:#deccelerating zone.
                omega = self._omega_max_eef*(1-np.cos(math.pi/2.0*(angle/self._theta_deccelerate)))
            omega_transform = omega*rotvec_transform# calcualte the rotation velocity.
            return omega_transform
    
    def clip_rotation_vec(self,vec):
        """Clip a rotation vector between 0 and 2*pi."""
        norm = np.linalg.norm(vec)
        if norm < 1e-8:
            return np.zeros(3)
        elif (norm>=1e-8):
            norm_tmp = norm
            if (norm > 2.0 * math.pi):#clip rotation angle between 0 and 2.0*pi
                norm_tmp = math.fmod(norm, 2.0 * math.pi)
                vec = vec / norm * norm_tmp #norm : current rotation angle. norm_tmp : clipped angle between 0 and 2*math.pi.
                return vec
            else:
                return vec

    def calculate_rotation_angle_between_rotvecs(self,rotation_vector1, rotation_vector2):
        """Calculate rotation angle from two rotation vectors.

        Args:
            rotation_vector1 (numpy.ndarray): current rotation vector (3)
            rotation_vector2 (numpy.ndarray): target rotation vector (3)

        Returns:
            angle (float) : rotation angle
            rotation_transform (numpy.ndarray) : normalized transform rotation vector (3)
        """
        # Clip rotation vector between 0 and 2 pi.  Is this useful? 
        #rotation_vector1 = self.clip_rotation_vec(rotation_vector1)
        #rotation_vector2 = self.clip_rotation_vec(rotation_vector2)
        
        # Ensure numpy arrays of shape (3,1) for cv2.Rodrigues
        rotation_vector1 = np.asarray(rotation_vector1, dtype=np.float64).reshape(3, 1)
        rotation_vector2 = np.asarray(rotation_vector2, dtype=np.float64).reshape(3, 1)
        #print(f"rotation_vector1={rotation_vector1},rotation_vector2={rotation_vector2}")

        try:
            # Convert rotation vectors to rotation matrices
            R1, _ = cv2.Rodrigues(rotation_vector1)
            R2, _ = cv2.Rodrigues(rotation_vector2)
            
            # Check if Rodrigues conversion was successful
            if R1 is None or R2 is None:
                return 0.0, np.array([0.0, 0.0, 0.0])

            # Invert R1 (transpose works for rotation matrices)
            R1_inv = R1.T

            # Compute relative rotation matrix
            R12 = R2 @ R1_inv

            # Convert back to rotation vector
            rotation_transform, _ = cv2.Rodrigues(R12)
            
            # Check if Rodrigues conversion was successful
            if rotation_transform is None:
                return 0.0, np.array([0.0, 0.0, 0.0])

            #convert (3,1)>(3)
            rotation_transform = np.array(rotation_transform).squeeze() 
            #clip rotationo vector between 0 and 2pi.
            rotation_transform = self.clip_rotation_vec(rotation_transform)

            # Handle full rotation (2*pi)
            if abs(2 * np.pi - np.linalg.norm(rotation_transform)) < 1e-6:
                rotation_transform = np.zeros(3)
            
            angle = np.linalg.norm(rotation_transform) #calculate the rotation angle in radian.

            #print(f"rotataion_transform={rotation_transform},shape={rotation_transform.shape}")
            #normalize transform rotation vector.
            if angle > 1e-8:  # Avoid division by zero
                # Ensure rotation_transform is a numpy array and has the right shape
                rotation_transform = np.asarray(rotation_transform).flatten()
                rotation_transform = np.array([rotation_transform[0]/angle,rotation_transform[1]/angle,rotation_transform[2]/angle])
            else:
                rotation_transform = np.array([0.0, 0.0, 0.0])

            return angle, rotation_transform
            
        except (cv2.error, ValueError, np.linalg.LinAlgError):
            # If any error occurs in the rotation calculations, return default values
            return 0.0, np.array([0.0, 0.0, 0.0])

    def quaternion_to_rotvec(self, q):
        """
        Convert a unit quaternion to a rotation vector (axis-angle).
        q.x=rx*sin(theta/2),q.y=ry*sin(theta/2),q.z=rz*sin(theta/2),qw=cos(theta/2)

        Parameters:
            q : array-like of shape (4,)
                The quaternion [x, y, z, w]
        
        Returns:
            rotvec : ndarray of shape (3,)
                The rotation vector
        """
        q = np.array(q, dtype=float)
        if q.shape != (4,):
            raise ValueError("Quaternion must be a 4-element vector [x, y, z, w]")

        x, y, z, w = q
        norm = np.linalg.norm(q)
        if not np.isclose(norm, 1.0):
            q /= norm
            x, y, z, w = q

        angle = 2 * np.arccos(w)
        sin_half_angle = np.sqrt(1 - w * w)

        if sin_half_angle < 1e-8:
            # When angle is close to 0, the axis is arbitrary
            return np.array([0.0, 0.0, 0.0])
        else:
            axis = np.array([x, y, z]) / sin_half_angle
            return axis #angle * axis

    def quaternion_to_forward_vector(self,quarternion):
        """Convert a quaternion to a forward vector (assuming forward is +Z)."""
        [qx, qy, qz, qw] = quarternion
        x = 2 * (qx*qz + qw*qy)
        y = 2 * (qy*qz - qw*qx)
        z = 1 - 2 * (qx*qx + qy*qy)
        return np.array([x, y, z])

    def vector_angle(self,v1, v2):
        """Calculate the cosine value from the two rotatio vectors. """
        v1_norm = v1 / (np.linalg.norm(v1)+1e-15) #normalization
        v2_norm = v2 / (np.linalg.norm(v2)+1e-15) #normalization
        dot_product = np.clip(np.dot(v1_norm, v2_norm), -1.0, 1.0)
        return dot_product#np.arccos(dot_product)

    def evaluate_alignment(self,quaternion, direction_vector):
        """Evaluate the alignment between a quaternion and a direction vector.
        Parameters:
        -------------
        quaternion : np.array
            rotation vector (3)
        direction_vector : np.array
            ideal orientation (3)
        """
        forward_vector = quaternion.copy()#self.quaternion_to_forward_vector(quarternion=quaternion)
        
        angle = self.vector_angle(forward_vector, direction_vector)
        return angle #[rad]

    def quaternion_to_new_direction(self,quarternion, vx, vy, vz):
        """convert from the i

        Args:
            quarternion (_type_): _description_
            vx (_type_): _description_
            vy (_type_): _description_
            vz (_type_): _description_

        Returns:
            _type_: _description_
        """
        #Calculate the current forward direction
        current_forward = self.quaternion_rotate_vector(quarternion, 0, 0, 1)
        
        #Determine the rotation needed
        rotation_axis = np.cross(current_forward, [vx, vy, vz])
        rotation_axis_magnitude = np.linalg.norm(rotation_axis)
        
        if rotation_axis_magnitude < 1e-6:
            return quarternion  # No rotation needed
        
        rotation_axis = rotation_axis / rotation_axis_magnitude
        dot_product = np.dot(current_forward, [vx, vy, vz])
        angle = np.arccos(np.clip(dot_product, -1.0, 1.0))
        
        #Create a rotation quaternion
        half_angle = angle / 2
        sin_half_angle = np.sin(half_angle)
        qx2 = rotation_axis[0] * sin_half_angle
        qy2 = rotation_axis[1] * sin_half_angle
        qz2 = rotation_axis[2] * sin_half_angle
        qw2 = np.cos(half_angle)
        
        #Combine the rotations
        new_quarternion = self.quaternion_multiply(qx2, qy2, qz2, qw2, quarternion)
        
        return new_quarternion

    def quaternion_rotate_vector(self,quarternion, vx, vy, vz):
        qx=quarternion[0] 
        qy=quarternion[1] 
        qz=quarternion[2]
        qw=quarternion[3]

        t2 = qw * qx
        t3 = qw * qy
        t4 = qw * qz
        t5 = -qx * qx
        t6 = qx * qy
        t7 = qx * qz
        t8 = -qy * qy
        t9 = qy * qz
        t10 = -qz * qz
        
        rx = vx * (2 * (t8 + t10) + 1) + vy * 2 * (t6 - t4) + vz * 2 * (t7 + t3)
        ry = vx * 2 * (t6 + t4) + vy * (2 * (t5 + t10) + 1) + vz * 2 * (t9 - t2)
        rz = vx * 2 * (t7 - t3) + vy * 2 * (t9 + t2) + vz * (2 * (t5 + t8) + 1)
        
        return rx, ry, rz

    def quaternion_multiply(self,qx1, qy1, qz1, qw1,quarternion):
        qx2=quarternion[0]
        qy2=quarternion[1] 
        qz2=quarternion[2]
        qw2=quarternion[3]

        qx = qw1*qx2 + qx1*qw2 + qy1*qz2 - qz1*qy2
        qy = qw1*qy2 - qx1*qz2 + qy1*qw2 + qz1*qx2
        qz = qw1*qz2 + qx1*qy2 - qy1*qx2 + qz1*qw2
        qw = qw1*qw2 - qx1*qx2 - qy1*qy2 - qz1*qz2

        return [qx, qy, qz, qw]

    def calculate_jacobian(self):
        """Calculate the Jacobian matrix for the end-effector."""
        zero_vec = [0.0] * 6  # Assuming 6 joints
        joint_angles = [p.getJointState(self.ur5, i, physicsClientId=self.client_id)[0] for i in range(6)]
        # Calculate Jacobian at the end-effector's origin
        jac_t, jac_r = p.calculateJacobian(self.ur5, self.end_effector_index, [0,0,0],
                                          joint_angles, zero_vec, zero_vec, physicsClientId=self.client_id)
        jacobian = np.vstack((jac_t, jac_r))
        return np.array(jacobian)

    def rotation_vector_to_quaternion(self,rot_vec):
        """
        Convert rotation vector to quaternion.
        
        rot_vec: (3,) array-like, rotation vector (axis * angle)
        
        Returns:
            quat: (4,) numpy array, quaternion [x, y, z, w]
        """
        theta = np.linalg.norm(rot_vec)
        
        if theta < 1e-8:  # very small angle, avoid division by zero
            return np.array([0.0, 0.0, 0.0, 1.0])
        
        axis = rot_vec / theta
        half_theta = theta / 2.0
        sin_half_theta = np.sin(half_theta)
        
        qx = axis[0] * sin_half_theta
        qy = axis[1] * sin_half_theta
        qz = axis[2] * sin_half_theta
        qw = np.cos(half_theta)
        
        return np.array([qx, qy, qz, qw])

    def changeMotion(self):
        """Change the human's turning conditions"""
        variance_change = 0.02

        #slightly change the current moving center
        self.moving_point_center[0] += variance_change*np.random.uniform(-1,1)#x
        self.moving_point_center[1] += variance_change*np.random.uniform(-1,1)#y
        self.moving_point_center[2] += variance_change*np.random.uniform(-1,1)#z
        #limit the z-axis in the center point.
        self.moving_point_center[2] = min(self._z_ub,max(self._z_lb,self.moving_point_center[2]))
        
        #Determine the human movement radius.
        zmin=self._zmin_rob
        zmax=self._zmax_rob
        tmp_radius = self.moving_point_radius 
        tmp_radius += variance_change*np.random.uniform(-1,1) # get the randomized turning radius.
        while (zmin<=self._zmin_rob or self._zmax_rob<=zmax):#not meet the criteia.
            tmp_radius = self.moving_point_radius 
            tmp_radius += variance_change*np.random.uniform(-1,1) # get the randomized turning radius.
            zmin = self.moving_point_center[2]-tmp_radius
            zmax = self.moving_point_center[2]+tmp_radius
        self.moving_point_radius = tmp_radius
        self.moving_point_radius = max(self._r_min_turn*self.rope_length,min(self._r_max_turn*self.rope_length,self.moving_point_radius))  # get the randomized turning radius.

        #Human's turning point sampling.
        #Get the robot's end-effector position.
        pos_end_effector = self.pose_eef[:3]  # Get the position (x, y, z)
        vec_r2h = pos_end_effector-self.moving_point_center
        vec_r2h /= (np.linalg.norm(vec_r2h)+1e-9)#normalized vector.

        #calculate the unit vectors in the turning plane.
        vec_r2h_plane = vec_r2h.copy() #remove z-axis element.
        vec_r2h_plane[2] = 0.0
        vec_r2h_plane /= np.linalg.norm(vec_r2h_plane)
        u1,u2 = self.compute_orthogonal_vectors(vec_r2h_plane)
        #prepare all samples in the circular trajectory. (360,3(x,y,z))
        self.samples_turn = np.array([[self.moving_point_center[j]+self.moving_point_radius*(np.cos(i/180*math.pi)*u1[j]+np.sin(i/180*math.pi)*u2[j]) for j in range(3)] for i in range(360)])#sample from the every angle. 

        #step6. human's turning speed
        #omega_h = np.random.uniform(self._v_hum_lb/self.moving_point_radius,self._v_hum_ub/self.moving_point_radius)#[rad/s]. v=r*omega. randomize the angular velocity.
        omega_h = np.random.uniform(self._omega_hum_lb,self._omega_hum_ub)
        self.deg_per_sec = min(int(360*self._n_max_turn),omega_h*180/math.pi) #[degree/s]

        dist_cur2sample = 1e5
        p_current = self.target_position
        for j in range(self.samples_turn.shape[0]):
            p_sample = self.samples_turn[j]
            d = np.linalg.norm(p_sample-p_current)
            if d<dist_cur2sample:
                dist_cur2sample=d
                self.n_degree = j

        self.target_position = self.samples_turn[self.n_degree]
        self.target_angle = 0.0

        #for motion-policy 2, make the corresponding velocity list.
        #1. check the position when the sign for the change in z-ccordinate switches.
        idx_switch = 0
        for i in range(1,self.samples_turn.shape[0]):#for each samples.
            dz_past = self.samples_turn[i][2]-self.samples_turn[i-1][2] #z-coordinate
            dz_next = self.samples_turn[(i+1)%self.samples_turn.shape[0]][2]-self.samples_turn[i][2]
            if dz_past>=0.0:#change in deltaz. positive
                if dz_next<=0.0:#non-positive
                    idx_switch=i
                    break
        
        #idx_switch is the baseline to determine the velocity.
        idx_st_accelerate = idx_switch-90 #the slowes point
        idx_end_accelerate = (idx_st_accelerate+179)%360 #the fastest point.
        omega_max = self.deg_per_sec*(1.0+self._variance_acc) #[degree/s]
        omega_min = self.deg_per_sec*(1.0-self._variance_acc) #[degree/s]
        self.vel_samples_turn = omega_min*np.ones(360)#[deg/s]. make an array with (360,)
        acc_per_angle = (omega_max-omega_min)/180.0
        bool_accelerate = True
        counter = 0
        idx = idx_st_accelerate

        self.n_circuit = 0

        #make self.vel_samples_turn. start from the slowest point.
        while (counter<=self.samples_turn.shape[0]):#for each samples.
            #iterator : counter, idx.
            if counter==180:
                bool_accelerate=False
            if bool_accelerate:#accelerating zone
                self.vel_samples_turn[idx] = self.vel_samples_turn[idx-1]+acc_per_angle#increment by acc_per_angle.
            else:#deccelerating zone.
                self.vel_samples_turn[idx] = self.vel_samples_turn[idx-1]-acc_per_angle#increment by acc_per_angle.
            #increment counter
            counter+=1
            idx = (idx+1)%360
        
        #reload lope
        #self.load_rope()
        
        #debug
        if self._bool_debug:
            fig2,ax2 = plt.subplots(2,1,figsize=(10,15))
            ax2[0].plot(np.arange(0,360),self.samples_turn[:,2])
            ax2[1].plot(np.arange(0,360),self.vel_samples_turn)

            #robot, human and turning samples configuration.
            #prepare figures
            fig3,ax3 = plt.subplots(2,2,figsize=(16,16),subplot_kw={'projection': '3d'})
            for j in range(4):
                # Customize the plot
                ax3[j//2,j%2].set_xlabel('X [m]')
                ax3[j//2,j%2].set_ylabel('Y [m]')
                ax3[j//2,j%2].set_zlabel('Z [m]')
                ax3[j//2,j%2].set_xlim(-1.5,1.0)
                ax3[j//2,j%2].set_ylim(-1.0,1.5)
                ax3[j//2,j%2].set_zlim(0,1)
            ax3[0,1].view_init(elev=0, azim=90) # from X
            ax3[1,0].view_init(elev=0, azim=0) # from Y
            ax3[1,1].view_init(elev=90, azim=-90) # from Z
            fig3.suptitle("Change motion debug")

            for i in range(4):
                #robot
                ax3[i//2,i%2].plot(pos_end_effector[0],pos_end_effector[1], pos_end_effector[2], color="lightblue",marker="o",markersize=15,label="robot")
                #turning center.
                ax3[i//2,i%2].plot(self.moving_point_center[0],self.moving_point_center[1], self.moving_point_center[2], color="r",marker="*",markersize=5,label="center")
                for j in range(self.samples_turn.shape[0]):
                    ax3[i//2,i%2].plot(self.samples_turn[j][0],self.samples_turn[j][1], self.samples_turn[j][2], color="r",marker="*",markersize=2)
            ax3[0,0].legend(fontsize=18)
            plt.show()
    
    def reset(self,i=0,bool_base=False,bool_eval=False):
        """Reset the environment for a new episode.
        bool_base: if False (for training), reset self.rosbag with the baseline model.
        if True (baseline), keep the self.rosbag to replayr the same situation.
        """

        #reset the simulation.
        p.resetSimulation(physicsClientId=self.client_id)
        p.setGravity(0, 0, -9.8, physicsClientId=self.client_id)
        p.setTimeStep(self.dt_sample, physicsClientId=self.client_id)  # Set the control frequency.
        #plane
        self.plane = p.loadURDF(self.PLANE, physicsClientId=self.client_id)

        if not bool_base: #for training : reset the dictionary.
            self.rosbag = dict() #reset the dictionary.
            self.bool_base = bool_base
        else: #for baseline : keep the dictionary.
            self.bool_base = bool_base

        #change friction
        self.bool_eval = bool_eval
        if not bool_base: #for training : change the friction.:
            lateralFriction = np.random.uniform(0.2,1.0)
            spinningFriction = np.random.uniform(0.0005,0.01)
            rolliingFriction= np.random.uniform(0.0005,0.01)		
            if bool_eval: #for evaluation.
                lateralFriction = 0.5
                rollingFriction = 0.01
                spinningFriction = 0.01
                
            self.rosbag["lateralFriction"] = lateralFriction
            self.rosbag["spinningFriction"] = spinningFriction
            self.rosbag["rolliingFriction"] = rolliingFriction
        else: #for baseline : keep the friction.
            lateralFriction = self.rosbag["lateralFriction"]
            spinningFriction = self.rosbag["spinningFriction"]
            rolliingFriction = self.rosbag["rolliingFriction"]
        
        p.changeDynamics(
            self.plane,
            linkIndex=-1, #base index
            lateralFriction=lateralFriction, #default is 0.5
            spinningFriction=spinningFriction,
            rollingFriction=rolliingFriction,
              physicsClientId=self.client_id
        )
        ########################################################


        # Parameter initialization.
        #rope
        # Rope parameters
        self.rope_length = 0.5 #0.4 m -> 0.5 m
        self.num_links = 50
        self.link_length = self.rope_length / self.num_links
        self.link_mass = 0.01 #kg/m -> 1.0
        self.rope_link_ids = []
        self.target_robot=1e5*np.zeros((3,3)) #for saving target positions.
        self.counter_save_robot = 0
        self.prev_link_vel = 1e5 #previous link velocity. This is for estimating force and torque.
        self.rope_mid_point = np.zeros(3)

        #camera setting
        # Camera parameters
        self.camera_distance =1.2  # Distance from the target (3 meters)
        self.camera_pitch = 0.0     # Keep pitch at 0 degrees
        self.camera_yaw = 0.0       # Start yaw at 0 degrees
        self.camera_target = [0, 0, 0]  # Target point in world coordinates

        #UR robot.
        self.config_robot = np.zeros((6,6))#6 joints' 6D pose. (x,y,z,rx,ry,rz)
        self.pose_eef = np.zeros(6) #6D eef pose.
        self.joints_list = np.zeros(6) #robot's joint's angles. [rad]
        self.omega_joint = np.zeros(6) #joint's angular velocity.
        self.v_eef = np.zeros(6) #eef velocity.
        self.ur_pose_env=[]
        self.ur_pose_custom = []
        self.joint_storage = []

        # Assuming you have 6 joints for the UR5e
        self.joints_list = np.array([-math.pi,-0.314,-math.pi/2.0,-0.48,math.pi/2.0,0.0])  # Example joint angles (radians)
        self.joint_storage.append(self.joints_list.tolist())
        #2. calculate the whole robot's configuration.
        self._ur_ctrl.cal_pose_all(self.joints_list)
        #update end-effector pose.
        self.pose_eef = self._ur_ctrl.pose6
        #update robot's configuration.
        self.config_robot[0,:]=self._ur_ctrl.pose1
        self.config_robot[1,:]=self._ur_ctrl.pose2
        self.config_robot[2,:]=self._ur_ctrl.pose3
        self.config_robot[3,:]=self._ur_ctrl.pose4
        self.config_robot[4,:]=self._ur_ctrl.pose5
        self.config_robot[5,:]=self._ur_ctrl.pose6
        #update robot's rendering.
        self.make_robot_skeleton(joint_positions=self.config_robot)

        #storage
        if not self.bool_base: #for training : reset time_step
            self.time_step = 1
        else: #for baseline : reset time_step to start from beginning
            self.time_step = 1
        self.samples_turn = []#(360,3(x,y,z))
        self.vel_samples_turn = []#(360,) : How many steps to proceed if the wrist is in the correspondent 3D position.
        self.deg_per_sec = 0
        self.R_r2h=0
        self.n_circuit = 0 #How many circuits
        self.n_degree = 0 #How many degree the human's wrist proceeds.
        self.pos_human_current = np.array([0,0,0])#(3,): current human's wrist position. [x,y,z]
        self.pos_human_prev = np.array([0,0,0])#(3,)
        self.vel_human = np.array([0,0,0])#(3,). human's wrist speed. [vx,vy,vz]

        self.r_current=-1
        self.r_previous=-1
        self.center_prev = np.zeros(4) #previous fitting center. (3)
        self.middle_rope_list = [] #storage for saving middle points
        self.prev_link_vels = {}

        #if (len(self.ur_pose_env)>100):
        #    fig,ax = plt.subplots(1,6,figsize=(30,6))
        #    labels=["x","y","z","rx","ry","rz"]
        #    time_scale = np.linspace(self.dt_sample,len(self.ur_pose_custom)*self.dt_sample,len(self.ur_pose_custom)) #start,stop,length
        #    for i in range(6):
        #        ax[i].plot(time_scale,np.array(self.ur_pose_env)[:,i],linestyle="--",linewidth=2,color="k",alpha=0.8,label="env")
        #        ax[i].plot(time_scale,np.array(self.ur_pose_custom)[:,i],linestyle="-",linewidth=2,color="r",label="custom")
        #        ax[i].set_title(labels[i],fontsize=16)
        #    ax[0].legend(fontsize=16)
        #    fig.suptitle("Joint angle transition")
        #################################################


        #Initialize the rope properties. ################
        #>>1. rope densitity: M [kg/m]
        #>>2. rope length:    L [m]
        if self.bool_base: #for baseline : keep the rope properties.
            self.link_mass = self.rosbag["link_mass"]
            self.rope_length = self.rosbag["rope_length"]
            self.link_length = self.rosbag["link_length"]
            self.num_links = self.rosbag["num_links"]

        else: #for training : randomize the rope properties.    
            #1. Rope mass
            self.link_mass = np.random.uniform(self._m_rope_lb, self._m_rope_ub) #kg/link        
            #2. Rope length
            self.rope_length = np.random.uniform(self._l_rope_lb, self._l_rope_ub)
            self.link_length = 0.01 # each link's length is 1 cm.
            self.num_links = int (self.rope_length/self.link_length)
            if bool_eval: #evaluation
                self.link_mass = 5.67e-4
                rope_length_list = [1.0,2.0,3.0]
                self.rope_length = rope_length_list[i%len(rope_length_list)]
                self.link_length = 0.01 # each link's length is 1 cm.
                self.num_links = int (self.rope_length/self.link_length)
            self.rosbag["link_mass"] = self.link_mass
            self.rosbag["rope_length"] = self.rope_length
            self.rosbag["link_length"] = self.link_length
            self.rosbag["num_links"] = self.num_links

        self.target_robot=1e5*np.zeros((3,3)) #for saving target positions.
        self.counter_save_robot = 0
        self.prev_link_vel = 1e5 #previous link velocity. This is for estimating force and torque.
        #################################################

        #Initialize the human turning condition #########
        #>> 1. turning center (xh,yh,zh)
        #>> 2. turning radius (rh)
        #>> 3. turning motion : (A) constant velocity (omega_h). (B) epsilon-greedy action selection. (C) designated human motion.
        #>>     (B) prepare N samples and if np.random()>epsilon(=0.1): omega_h, else: np.arange(point_per_frame*(1-a),point_per_frame*(1+a))  point_per_frame [degree/frame] = omega_h [rad/s] /freq [frame/s] *180/math.pi [degree/rad]
        #>>     (C) expertise from the actual human motion.
        # Target movement parameters.
        
        #step1. Get the robot's end-effector position.
        pos_end_effector = self.pose_eef[:3].copy()

        #step2 calculate the distance between human and robot.
        low = self._rate_h2r_lb * self.rope_length
        high = self._rate_h2r_ub * self.rope_length
        self.R_r2h = np.random.uniform(low, high)
        self.dist_r2h_current = self.R_r2h

        if bool_eval:
            self.R_r2h = 0.707*self.rope_length
            self.dist_r2h_current = self.R_r2h

        #step3. decide the human's turning center
        self.moving_point_center = [0,0,0] #0.6
        vec_r2h = np.ones(3)
        while (vec_r2h[0]>=0.0 or self.moving_point_center[2]<self._z_lb or self._z_ub<self.moving_point_center[2]):
            vec_r2h = np.array([np.random.uniform(-1,0.0),np.random.uniform(-1,1),np.random.uniform(-1,1)])
            if bool_eval:
                vec_r2h = np.array([-1.0,0.0,0.0])
            vec_r2h = vec_r2h/np.linalg.norm(vec_r2h+1e-10) #normalize the vector.
            self.moving_point_center = pos_end_effector + self.R_r2h*vec_r2h
        
        #step4. Determine the human movement radius.
        zmin=self._zmin_rob
        zmax=self._zmax_rob
        self.moving_point_radius = np.random.uniform(self._r_min_turn*self.rope_length,self._r_max_turn*self.rope_length)  # get the randomized turning radius.
        if bool_eval:
            self.moving_point_radius = 0.15*self.rope_length
        else:
            while (zmin<=self._zmin_rob or self._zmax_rob<=zmax):#not meet the criteia.
                self.moving_point_radius = np.random.uniform(self._r_min_turn*self.rope_length,self._r_max_turn*self.rope_length)  # get the randomized turning radius.
                zmin = self.moving_point_center[2]-self.moving_point_radius
                zmax = self.moving_point_center[2]+self.moving_point_radius
        
        #step5. Human's turning point sampling.
        #calculate the unit vectors in the turning plane.
        vec_r2h_plane = vec_r2h.copy() #remove z-axis element.
        vec_r2h_plane[2] = 0.0
        vec_r2h_plane /= np.linalg.norm(vec_r2h_plane)
        u1,u2 = self.compute_orthogonal_vectors(vec_r2h_plane)

        #prepare all samples in the circular trajectory. (360,3(x,y,z))
        self.samples_turn = np.array([[self.moving_point_center[j]+self.moving_point_radius*(np.cos(i/180*math.pi)*u1[j]+np.sin(i/180*math.pi)*u2[j]) for j in range(3)] for i in range(360)])#sample from the every angle. 
        self.target_position = self.samples_turn[0]
        self.target_angle = 0.0

        if not self.bool_base: #for training : save the human's turning condition.
            self.rosbag["target_position"] = np.array([self.target_position])    #(1,3)        
        else:#for baseline : keep the human's turning condition.
            self.target_position = self.rosbag["target_position"][0]
        self.counter_step = 1

        #step6. human's turning speed
        #omega_h = np.random.uniform(self._v_hum_lb/self.moving_point_radius,self._v_hum_ub/self.moving_point_radius)#[rad/s]. v=r*omega. randomize the angular velocity.
        omega_h = np.random.uniform(self._omega_hum_lb,self._omega_hum_ub)
        if bool_eval:
            omega_h = (self._omega_hum_lb+self._omega_hum_ub)/2.0
        self.deg_per_sec = min(int(360*self._n_max_turn),omega_h*180/math.pi) #[degree/s]

        #for motion-policy 2, make the corresponding velocity list.
        #1. check the position when the sign for the change in z-ccordinate switches.
        idx_switch = 0
        for i in range(1,self.samples_turn.shape[0]):#for each samples.
            dz_past = self.samples_turn[i][2]-self.samples_turn[i-1][2] #z-coordinate
            dz_next = self.samples_turn[(i+1)%self.samples_turn.shape[0]][2]-self.samples_turn[i][2]
            if dz_past>=0.0:#change in deltaz. positive
                if dz_next<=0.0:#non-positive
                    idx_switch=i
                    break
        
        #idx_switch is the baseline to determine the velocity.
        idx_st_accelerate = idx_switch-90#the slowes point
        idx_end_accelerate = (idx_st_accelerate+179)%360 #the fastest point.
        omega_max = self.deg_per_sec*(1.0+self._variance_acc) #[degree/s]
        omega_min = self.deg_per_sec*(1.0-self._variance_acc) #[degree/s]
        self.vel_samples_turn = omega_min*np.ones(360)#[deg/s]. make an array with (360,)
        acc_per_angle = (omega_max-omega_min)/180.0
        bool_accelerate = True
        counter = 0
        idx = idx_st_accelerate

        #make self.vel_samples_turn. start from the slowest point.
        while (counter<=self.samples_turn.shape[0]):#for each samples.
            #iterator : counter, idx.
            if counter==180:
                bool_accelerate=False
            if bool_accelerate:#accelerating zone
                self.vel_samples_turn[idx] = self.vel_samples_turn[idx-1]+acc_per_angle#increment by acc_per_angle.
            else:#deccelerating zone.
                self.vel_samples_turn[idx] = self.vel_samples_turn[idx-1]-acc_per_angle#increment by acc_per_angle.
            #increment counter
            counter+=1
            idx = (idx+1)%360
        
        #load rope
        self.load_rope()

        #camera position
        #robot's end-effector.
        p_eef = self.pose_eef[:3].copy()  # Get the position (x, y, z)
        #human's turning center
        p_h = np.array(self.moving_point_center) #(x,y,z)
        #vector from the robot to human
        vec_eef2h = p_h - p_eef
        dist_eef2h = np.linalg.norm(vec_eef2h)
        u_eef2h = vec_eef2h/dist_eef2h #unit vector.
        self.p_cam_goal = (p_h+p_eef)/2.0
        #rotate 30 degree.
        theta = np.pi/5
        scale = 3.0
        if not self.bool_base: #for training : save the camera position.
            self.x_cam = p_eef[0] + (u_eef2h[0]*np.cos(theta)-u_eef2h[1]*np.sin(theta)) * scale
            self.y_cam = p_eef[1] + (u_eef2h[1]*np.cos(theta)+u_eef2h[0]*np.sin(theta)) * scale
            self.z_cam = self._zmax_rob
            self.rosbag["x_cam"] = self.x_cam
            self.rosbag["y_cam"] = self.y_cam
            self.rosbag["z_cam"] = self.z_cam
        else:
            self.x_cam = self.rosbag["x_cam"]
            self.y_cam = self.rosbag["y_cam"]
            self.z_cam = self.rosbag["z_cam"]
            
        #debug
        if self._bool_debug:
            fig2,ax2 = plt.subplots(2,1,figsize=(10,15))
            ax2[0].plot(np.arange(0,360),self.samples_turn[:,2])
            ax2[1].plot(np.arange(0,360),self.vel_samples_turn)

            #robot, human and turning samples configuration.
            #prepare figures
            fig3,ax3 = plt.subplots(2,2,figsize=(16,16),subplot_kw={'projection': '3d'})
            for j in range(4):
                # Customize the plot
                ax3[j//2,j%2].set_xlabel('X [m]')
                ax3[j//2,j%2].set_ylabel('Y [m]')
                ax3[j//2,j%2].set_zlabel('Z [m]')
                ax3[j//2,j%2].set_xlim(-1.5,1.0)
                ax3[j//2,j%2].set_ylim(-1.0,1.5)
                ax3[j//2,j%2].set_zlim(0,1)
            ax3[0,1].view_init(elev=0, azim=90) #from x
            ax3[1,0].view_init(elev=0, azim=0)#from Y
            ax3[1,1].view_init(elev=90, azim=-90)#from z

            for i in range(4):
                #robot
                ax3[i//2,i%2].plot(pos_end_effector[0],pos_end_effector[1], pos_end_effector[2], color="lightblue",marker="o",markersize=15,label="robot")
                #turning center.
                ax3[i//2,i%2].plot(self.moving_point_center[0],self.moving_point_center[1], self.moving_point_center[2], color="r",marker="*",markersize=5,label="center")
                for j in range(self.samples_turn.shape[0]):
                    ax3[i//2,i%2].plot(self.samples_turn[j][0],self.samples_turn[j][1], self.samples_turn[j][2], color="r",marker="*",markersize=2)
            ax3[0,0].legend(fontsize=18)
            plt.show()

        #################################################
        
        obs = self._get_observation()
        #normalize only the norm. Adopt the sign separately.
        obs_normalized = obs.copy() #copy the observation.
        #position.
        obs_normalized[:3] = np.array([np.sign(e)*(abs(e)-p_min)/(p_max-p_min) for e,p_min,p_max in zip(obs[:3],self._p_min,self._p_max)])
        #rotation vector.
        obs_normalized[3:6] = np.array([np.sign(q)*(abs(q)-0.0)/(self._q_max-0.0) for q in obs[3:6]])
        #end-effector linear velocity. (vx,vy,vz,wx,wy,wz)
        obs_normalized[6:9] = np.array([np.sign(v)*(abs(v)-0.0)/(self._v_max-0.0) for v in obs[6:9]])
        #end-effector angular velocity.
        obs_normalized[9:12] = np.array([np.sign(w)*(abs(w)-0.0)/(self._omega_max-0.0) for w in obs[9:12]])
        #human's wrist position
        obs_normalized[12:15] = np.array([np.sign(e)*(abs(e)-p_min)/(p_max-p_min) for e,p_min,p_max in zip(obs[12:15],self._p_min,self._p_max)])
        #human's linear velocity. (vx,vy,vz)
        obs_normalized[15:18] = np.array([np.sign(v)*(abs(v)-0.0)/(self._v_max-0.0) for v in obs[15:18]])
        #force
        obs_normalized[18:21] = np.array([np.sign(f)*(abs(f)-self._force_min)/(self._force_max-self._force_min) for f in obs[18:21]])
        #torque
        obs_normalized[21:24] = np.array([np.sign(t)*(abs(t)-self._torque_min)/(self._torque_max-self._torque_min) for t in obs[21:24]])
        #rope state
        #rope length
        obs_normalized[24] = obs[24]/self._l_rope_max
        #rope middle position.
        obs_normalized[25:28] = np.array([np.sign(p)*(abs(p)-p_min)/(p_max-p_min) for p,p_min,p_max in zip(obs[25:28],self._p_min,self._p_max)])
        #rope middle velocity.
        obs_normalized[28:31] = np.array([np.sign(v)*(abs(v)-0.0)/(self._v_max-0.0) for v in obs[28:31]])

        return obs_normalized

    def compute_orthogonal_vectors(self, v):
        """Calculate the orghogonal two vectors from the norm vecotr.

        Args:
            v (numpy.ndarray): norm vector

        Returns:
            u1, u2: two orthogonal vecotrs in the perpendicular plane to v.
        """
        v = np.asarray(v, dtype=float)
        v = v / np.linalg.norm(v)  # Normalize input vector

        # Choose a vector not parallel to v
        if abs(v[0]) < abs(v[1]) and abs(v[0]) < abs(v[2]):
            temp = np.array([1.0, 0.0, 0.0])
        elif abs(v[1]) < abs(v[2]):
            temp = np.array([0.0, 1.0, 0.0])
        else:
            temp = np.array([0.0, 0.0, 1.0])

        # First orthogonal vector (cross product)
        u1 = np.cross(v, temp)
        u1 /= np.linalg.norm(u1)  # Normalize

        # Second orthogonal vector (cross product)
        u2 = np.cross(v, u1)
        u2 /= np.linalg.norm(u2)  # Normalize

        return u1, u2

    def render(self, mode='human'):
        if mode == 'rgb_array':
            # Move the camera 90 degrees in yaw
            self.camera_yaw = 90.0  # Update yaw to 90 degrees

            # Calculate new camera position
            camera_pos = [self.x_cam,self.y_cam,self.z_cam]#self.get_camera_position()

            # Use pybullet's camera to capture an RGB image from the environment
            width, height, rgb_img, _, _ = p.getCameraImage(640, 480,
                                                            viewMatrix=p.computeViewMatrix(
                                                                cameraEyePosition=camera_pos,
                                                                cameraTargetPosition=self.p_cam_goal,
                                                                cameraUpVector=[0, 0, 1]  # Assuming Z is up
                                                            , physicsClientId=self.client_id
                                                            ),
                                                            projectionMatrix=p.computeProjectionMatrixFOV(
                                                                fov=90,  # Field of view
                                                                aspect=640 / 480,
                                                                nearVal=0.1,
                                                                farVal=20, 
                                                                physicsClientId=self.client_id
                                                            ), physicsClientId=self.client_id)
            # Convert image to a NumPy array and reshape it to (height, width, 4) (RGBA format)
            rgb_img = np.array(rgb_img, dtype=np.uint8)
            rgb_img = rgb_img.reshape((height, width, 4))
            
            # Remove the alpha channel (RGBA -> RGB)
            rgb_img = rgb_img[:, :, :3]
            #print("rgb_img.shape=",rgb_img.shape)
            if not isinstance(rgb_img, np.ndarray):
                rgb_img = np.array(rgb_img)
            return rgb_img
        else:
            pass#self.env.render(mode)

    def close(self):
        """Disconnect from PyBullet."""
        p.disconnect(physicsClientId=self.client_id)

    # Function to calculate camera position based on distance, pitch, and yaw
    def get_camera_position(self):
        pitch_rad = math.radians(self.camera_pitch)
        yaw_rad = math.radians(self.camera_yaw)

        x = -self.camera_distance#self.camera_distance * math.cos(pitch_rad) * math.sin(yaw_rad)
        y = self.camera_distance #* math.cos(pitch_rad) * math.cos(yaw_rad)
        z = 1.0#self.camera_distance * math.sin(pitch_rad)


        return [x, y, z]
