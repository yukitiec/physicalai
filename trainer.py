import time
import numpy as np
import torch
from datetime import timedelta
import os
import gym
import glob
from base64 import b64encode
from IPython.display import HTML
from gym.wrappers.monitoring import video_recorder
import cv2
import matplotlib.pyplot as plt
import pandas as pd
import time

"""CHECK HERE"""
rootDir = r"C:\Users\kawaw\python\pybullet\ur5-bullet\UR5\code\robot_RL_curriculum\imitation_learn"
type_reward = "ppo"
lr_type="constant"
bool_residual = False #Redidual learning or not.
file_base = os.path.join(rootDir,"ppo_actor_critic.pt") #Path to the baseline model. 
"""CHECK HERE"""

class Trainer: 

    def __init__(self, env, env_test,env_render, algo, 
                 bool_sac,fps=200,seed=0, num_steps=10**6, 
                 eval_interval=10**4, num_eval_episodes=3,step_epsiode=4*10**3,
                 rootDir=rootDir,type_reward=type_reward,lr_type=lr_type,saveDir=rootDir):
        
        self.env = env
        self.env_test = env_test
        self.env_render = env_render
        self.algo = algo
        self.bool_sac = bool_sac #True: SAC, False:PPO
        self.fps = fps

        self.saveDir = saveDir
        os.makedirs(self.saveDir, exist_ok=True)
        self.file_base = os.path.join(self.saveDir,"ppo_actor_critic.pt")

        # Set environment seeds for reproducibility
        #self.env.seed(seed)
        self.env_test.seed(2**31 - seed)

        # Dictionary to store average returns
        self.returns = {'step': [], 'return': []}

        # Training parameters
        self.num_steps = num_steps
        self.eval_interval = eval_interval
        self.num_eval_episodes = num_eval_episodes
        self.step_epsiode = step_epsiode

        #storage for alpha
        self.alpha_values = []
        #storage for learning rate
        self.lr_actor_list = []
        self.lr_critic_list = []

    def train(self):
        """Train the agent for a specified number of steps."""
        self.start_time = time.time()
        t = 0  # Episode step count

        # Reset the environment to get the initial state
        state = self.env.reset()

        # Initialize a variable to store the best validation loss
        best_val_reward = -1e10  # Or use a large value initially

        for steps in range(1, self.num_steps + 1):#for each step.
            if self.bool_sac:#SAC
                state, t,done = self.algo.step(self.env, state, t, steps)
            else:#PPO
                # Interact with the environment
                state, t, done = self.algo.step(self.env, state, t, steps)

            # If an episode has ended, pause for 20 seconds
            if done:
                state = self.env.reset()
                print("Episode ended.")
                
            # Update policy if enough steps have been collected
            if self.algo.is_update(steps):
                self.algo.update(num_step=steps)
                #save the current learning rate
                #Critic
                current_lr_critic = self.algo.optim_critic.param_groups[0]['lr']
                self.lr_critic_list.append(current_lr_critic)
                #Actor
                current_lr_actor = self.algo.optim_actor.param_groups[0]['lr']
                self.lr_actor_list.append(current_lr_actor)
                #print(f"current_lr_critic ={current_lr_critic},current_lr_actor ={current_lr_actor}")
                #entropy coefficient
                if self.bool_sac: #if SAC is being used
                    self.alpha_values.append(self.algo.alpha.item())
                

            # Periodically evaluate the policy
            if steps % self.eval_interval == 0:
                self.evaluate(steps)
                if self.returns['return'][-1] >= best_val_reward: #The higher reward, the better.
                    best_val_reward = self.returns['return'][-1]
                    # Call this function when you want to save the models
                    self.save_model(self.file_base)
                

    def evaluate(self, steps):
        """Evaluate the agent's performance over multiple episodes."""
        returns = []
        for i in range(self.num_eval_episodes):
            state = self.env_test.reset(i,bool_eval = True)
            print(f"i = {i} evaluation : rope_length = {self.env_test.rope_length}, R_r2h = {self.env_test.R_r2h}, moving_point_radius = {self.env_test.moving_point_radius}")
            done = False
            episode_return = 0.0

            while not done:
                # Select action deterministically for evaluation
                action = self.algo.exploit(state)
                state,state_rope_middle, reward, done, _ = self.env_test.step(action)
                episode_return += reward

            returns.append(episode_return)

        # Calculate average return over evaluation episodes
        mean_return = np.mean(returns)
        epoch = int(steps/self.step_epsiode)
        self.returns['step'].append(steps)
        self.returns['return'].append(mean_return)

        print(f'Num steps: {steps:<6}   '
              f'Return: {mean_return:<5.1f}   '
              f'Time: {self.time}'
              )

    def save_model(self, filename):
        # Save both actor and critic models to a single file
        # Save the actor, critic, and optimizers
        torch.save({
            'actor': self.algo.actor.state_dict(),
            'critic': self.algo.critic.state_dict(),
            'optim_actor': self.algo.optim_actor.state_dict(),
            'optim_critic': self.algo.optim_critic.state_dict()
        }, filename)
    
    def visualize(self):
        """Visualize a single episode using the trained policy."""
        #env = gym.make(self.env.unwrapped.spec.id)
         # Ensure the video folder exists
        video_folder = os.path.join(self.saveDir,"video")
        
        if not os.path.exists(video_folder):
            os.makedirs(video_folder)

        #self.env = gym.wrappers.RecordVideo(self.env, saveDir,episode_trigger=lambda x: True)  # Save videos to the 'videos/' directory
        state = self.env_render.reset(i=1,bool_eval=True)
        done = False

        counter=0
        fps = self.fps
        period_save = int(fps/10)
        observations = np.empty((0, 31))
        # Initialize an empty numpy array
        times = []
        joints_storage = []
        target_pos = np.random.rand(6)
        i = 0
        t_elapsed = 0
        counter = 0
        while not done:
            # Select action deterministically
            t_start = time.time()
            action = self.algo.exploit(state)
            t_elapsed += time.time() - t_start
            counter += 1
            if counter%period_save==0:
                state,state_rope_middle, reward, done, _ = self.env_render.step(action)
            else:
                state,state_rope_middle, reward, done, _ = self.env_render.step(action)
            observations = np.append(observations,[state],axis=0)
            joints_storage.append(self.env_render.joints_list.tolist())
            times.append(counter/fps)
            if counter%period_save==0:
                frame = self.env_render.render(mode="rgb_array")
                if isinstance(frame, np.ndarray):
                    file_img = os.path.join(video_folder,f"{counter:05d}.png")
                    #print("frame.shape=",frame.shape)
                    cv2.imwrite(file_img,frame)
            counter+=1
        print(f"####  fps = {counter/t_elapsed} ####")
        joints_storage = np.array(joints_storage)
        self.env_render.close()
        times = np.array(times)
        self.plot_eval(times,observations,joints_storage)

    def plot(self):
        """Plot the training progress."""
        
        fig = plt.figure(figsize=(8, 6))
        plt.plot(self.returns['step'], self.returns['return'])
        plt.xlabel('Episode', fontsize=20)
        plt.ylabel('Return', fontsize=20)
        plt.tick_params(labelsize=18)
        #plt.title('Reward', fontsize=24)
        plt.tight_layout()
        file_img = os.path.join(self.saveDir,"reward.png")
        plt.savefig(file_img)
        plt.clf()

        data_save = []
        for i in range(len(self.returns['step'])):
            temp = []
            temp.append(self.returns['step'][i])
            temp.append(self.returns['return'][i])
            data_save.append(temp)
        data_save = np.array(data_save)
        df = pd.DataFrame(data_save)
        file_csv =os.path.join(self.saveDir,"reward.csv")
        df.to_csv(file_csv, index=False, header=False)  # Set header=True if you need column names

    def plot_lr(self):
        """ plot lr transition """
        plt.figure(figsize=(8, 6))
        plt.plot(range(0, len(self.lr_critic_list) * self.eval_interval, self.eval_interval), self.lr_critic_list,label=r"$lr_{critic}$",color="r",linewidth=2)
        plt.plot(range(0, len(self.lr_actor_list) * self.eval_interval, self.eval_interval), self.lr_actor_list,label=r"$lr_{actor}$",color="b",linewidth=2)
        plt.xlabel('Episode',fontsize=16)
        plt.ylabel('Learning rate',fontsize=16)
        #plt.title('Transition of Alpha During Training')
        plt.tick_params(labelsize=16)
        plt.legend(fontsize=16)
        plt.tight_layout()
        file_alpha = os.path.join(self.saveDir,"alpha.png")
        plt.savefig(file_alpha)
        plt.clf()

    def plot_alpha_transition(self):
        """ plot alpha transition """
        plt.figure(figsize=(8, 6))
        plt.plot(range(0, len(self.alpha_values) * self.eval_interval, self.eval_interval), self.alpha_values)
        plt.xlabel('Episode',fontsize=16)
        plt.ylabel('Alpha Value',fontsize=16)
        #plt.title('Transition of Alpha During Training')
        plt.tick_params(labelsize=16)
        plt.tight_layout()
        file_alpha = os.path.join(self.saveDir,"alpha.png")
        plt.savefig(file_alpha)
        plt.clf()

        #save alpha transition in .csv file
        epoch_list = np.arange(0, len(self.alpha_values) * self.eval_interval, self.eval_interval)
        alpha_list = []
        for i in range(len(self.alpha_values)):
            alpha_list.append([epoch_list[i],self.alpha_values[i]]) #[epoch, alpha]
        alpha_list = np.array(alpha_list)

        df = pd.DataFrame(alpha_list)
        file_csv =os.path.join(self.saveDir,"alpha.csv")
        df.to_csv(file_csv, index=False, header=False)  # Set header=True if you need column names
        
    def plot_eval(self,times,observations,joints_storage):
        """
        Parameters:
        -----------
        observations : 
            end_effector_position, #3
            end_effector_orientation, #4 (7)
            vel_eef, #7 (14)
            self.pos_human_current, #3 (17)
            self.vel_human, #3 (20)
            net_force, #3 (23)
            net_torque, #3 (26)
            state_rope_middle  #7. (length,pos,vel) (33)
        """
        #robot joints.
        fig0,ax0 = plt.subplots(1,6,figsize=(35,6)) #(w,h)
        titles=["base","shoulder","elbow","wrist1","wrist2","wrist3"]
        for i in range(6):
            ax0[i].plot(times,joints_storage[:,i],color="k")
            ax0[i].set_title(titles[i],fontsize=20)
            ax0[i].set_xlabel("Time [sec]",fontsize=16)
            ax0[i].set_ylabel("Angle [radian]",fontsize=16)
            ax0[i].tick_params(axis='both', labelsize=14)
            #ax0[i].set_ylim(-6.28,6.28)
        #fig0.suptitle("Joint angles",fontsize=20)
        plt.tight_layout()
        file_img = os.path.join(self.saveDir,"joints.png")
        plt.savefig(file_img)
        plt.clf()

        # Convert to DataFrame and save
        data_save = []
        for i in range(joints_storage.shape[0]):
            temp = [times[i]]
            for j in range(6):
                temp.append(joints_storage[i][j])
            data_save.append(temp)
        data_save = np.array(data_save)
        df = pd.DataFrame(data_save)
        file_csv =os.path.join(self.saveDir,"joints.csv")
        df.to_csv(file_csv, index=False, header=False)  # Set header=True if you need column names


        #robot position and target position
        fig,ax = plt.subplots(1,3,figsize=(21,7))
        titles=["X ","Y","Z"]
        axes = ["X [m]","Y [m]","Z [m]"]
        for i in range(3):
            ax[i].plot(times,observations[:,i+12],color="r",label="Human")
            ax[i].plot(times,observations[:,i],color="b",label="Robot")
            #ax[i].set_title(titles[i],fontsize=18)
            ax[i].set_xlabel("Time [s]",fontsize=16)
            ax[i].set_ylabel(axes[i],fontsize=16)
            ax[i].tick_params(axis='both', labelsize=16)
            if (i==2):
                ax[i].legend(loc="best",fontsize=16)
            plt.tight_layout()
        file_img = os.path.join(self.saveDir,"position_transition.png")
        fig.savefig(file_img)
        plt.clf()
        #fig.show()

        data_save = []
        for i in range(observations.shape[0]):#for each sequence.
            temp = [times[i]]
            for j in range(3):#target position
                temp.append(observations[i][j])
            for j in range(3):#robot end-effector position
                temp.append(observations[i][3+j])
            data_save.append(temp)
        data_save = np.array(data_save)
        df = pd.DataFrame(data_save)
        file_csv =os.path.join(self.saveDir,"ee_transition.csv")
        df.to_csv(file_csv, index=False, header=False)  # Set header=True if you need column names

        distances = []
        for i in range(observations.shape[0]):
            d = ((observations[i,0]-observations[i,12])**2+(observations[i,1]-observations[i,13])**2+(observations[i,2]-observations[i,14])**2)**(0.5)
            distances.append(d)
        distances = np.array(distances)

        fig = plt.figure(figsize=(8, 6))
        plt.plot(times,distances,color="k",linewidth=2)
        plt.xlabel('Time [s]', fontsize=18)
        plt.ylabel('Distance [m]', fontsize=18)
        plt.axhline(y=self.env_render.rope_length, color='r', linestyle='--',linewidth=2)
        plt.tick_params(labelsize=16)
        #plt.title('distance between end effector and target', fontsize=18)
        plt.tight_layout()
        file_img = os.path.join(self.saveDir,"distance.png")
        plt.savefig(file_img)
        plt.clf()

        data_save = []
        for i in range(distances.shape[0]):#for each sequence
            temp = []
            temp.append(times[i])
            temp.append(distances[i])
            data_save.append(temp)
        data_save = np.array(data_save)
        df = pd.DataFrame(data_save)
        file_csv =os.path.join(self.saveDir,"distance.csv")
        df.to_csv(file_csv, index=False, header=False)  # Set header=True if you need column names

        #middle points
        #load ground truth 
        #file_path = r"C:\Users\kawaw\python\pybullet\ur5-bullet\UR5\code\robot_RL\middle_points.csv"
        #data = pd.read_csv(filepath_or_buffer=file_path)
        #data = data.values
        fig2,ax2 = plt.subplots(1,3,figsize=(21,7))
        titles=["X","Y","Z"]
        axes = ["X [m]","Y [m]","Z [m]"]
        for i in range(3):
            #ax2[i].plot(data[:,0],data[:,(i+1)],color="b",label="ideal")#time, (x,y,z)
            ax2[i].plot(times,observations[:,-6+i],color="k",label="middle")
            ax2[i].plot(times,observations[:,-13+i],color="r",label="force")
            ax2[i].set_title(titles[i],fontsize=16)
            ax2[i].set_xlabel("Time [s]",fontsize=16)
            ax2[i].set_ylabel(axes[i],fontsize=16)
            ax2[i].tick_params(axis='both', labelsize=16)
            if i==2:
                ax2[i].legend(loc="best",fontsize=16)
        #fig2.suptitle("Middle position of the long rope")
        plt.tight_layout()
        file_img = os.path.join(self.saveDir,"middle_point.png")
        fig2.savefig(file_img)
        plt.clf()

        # data_save = []
        # for i in range(data.shape[0]):#for each sequence.
        #     temp = []
        #     for j in range(4):
        #         temp.append(data[i,j])#add ideal data
        #     data_save.append(temp)
        # data_save = np.array(data_save)
        # df = pd.DataFrame(data_save)
        # file_csv =os.path.join(self.saveDir,"ee_ideal.csv")
        # df.to_csv(file_csv, index=False, header=False)  # Set header=True if you need column names

        data_save = []
        for i in range(observations.shape[0]):#for each sequence.
            temp = [times[i]]
            for j in range(3):
                temp.append(observations[i,-6+j])#add ideal data
            data_save.append(temp)
        data_save = np.array(data_save)
        df = pd.DataFrame(data_save)
        file_csv =os.path.join(self.saveDir,"middle.csv")
        df.to_csv(file_csv, index=False, header=False)  # Set header=True if you need column names

        data_save = []
        for i in range(observations.shape[0]):#for each sequence.
            temp = [times[i]]
            for j in range(3):
                temp.append(observations[i,-13+j])#add ideal data
            data_save.append(temp)
        data_save = np.array(data_save)
        df = pd.DataFrame(data_save)
        file_csv =os.path.join(self.saveDir,"force.csv")
        df.to_csv(file_csv, index=False, header=False)  # Set header=True if you need column names

    @property
    def time(self):
        """Calculate the elapsed training time."""
        return str(timedelta(seconds=int(time.time() - self.start_time)))
