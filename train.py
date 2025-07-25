import gym
import torch
import glob
import cv2
import os
import re
import pybullet as p

import ur5e_rope as ur5e
import sac_withppo as sac
import trainer 
import cosine_scheduler as lr_scheduler
import time
from datetime import datetime
def get_day_time():
    """Get the current day time.
    
    Returns:
        str: Current day time in format 'YYYY-MM-DD HH:MM:SS'
    """
    return datetime.now().strftime('%Y%m%d_%H%M%S')

# Define your environment (ensure UR5eRopeEnv is correctly implemented and accessible)
class UR5eRopeEnvWrapper(gym.Env):
    def __init__(self,fps,step_episode,client_id):
        self.env = ur5e.UR5eRopeEnv(fps=fps,step_episode=step_episode,client_id=client_id)
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        self.max_episode_steps = self.env.max_episode_steps
        self.joint_storage = self.env.joint_storage

    @property
    def normalizer(self):
        return self.env._normalizer

    @property
    def bool_base(self):
        """Get the current bool_base value from the underlying environment"""
        return self.env.bool_base

    @property #can be used as an attribute.
    def rosbag(self):
        """Get the current rosbag from the underlying environment"""
        return self.env.rosbag

    @property
    def joints_list(self):
        """Get the current joints_list from the underlying environment"""
        return self.env.joints_list
    
    @property	
    def rope_mid_point(self):
        return self.env.rope_mid_point
    
    @property
    def rope_mid_point_estimate(self):
        return self.env.rope_mid_point_estimate

    @property
    def rope_length(self):
        return self.env.rope_length
    
    @property
    def R_r2h(self):
        return self.env.R_r2h
    
    @property
    def moving_point_center(self):
        return self.env.moving_point_center
    
    @property
    def moving_point_radius(self):
        return self.env.moving_point_radius
    
    def step(self, action,bool_base=False,bool_render=False):
        return self.env.step(action,bool_base,bool_render)
    
    def step_demo(self):#for demonstration.
        return self.env.step_demo()

    def reset(self,i=0,bool_base=False,bool_eval=False):
        return self.env.reset(i,bool_base,bool_eval)

    def seed(self, seed=None):
        self.env.seed(seed)

    def render(self, mode='human'):
        if mode=="rgb_array":
            frame = self.env.render(mode)
            return frame
        else:
            self.env.render(mode)

    def close(self):
        self.env.close()

class MakeVideo():
    def __init__(self,fps,imgsize,src,videoname):
        self.fps = fps
        self.imgsize = imgsize
        self.src = src
        self.videoname = videoname
        self.main()

    def atoi(self,text):
        return int(text) if text.isdigit() else text

    def natural_keys(self,text):
        return [ self.atoi(c) for c in re.split(r'(\d+)', text) ]

    def main(self):
        img_array=[]
        for filename in sorted(glob.glob(f"{self.src}/*.png"), key=self.natural_keys):
            img = cv2.imread(filename)
            height, width, layers = img.shape
            size = (width, height)
            img_array.append(img)

        out = cv2.VideoWriter(self.videoname, cv2.VideoWriter_fourcc(*'mp4v'), self.fps, self.imgsize)#10fps

        for i in range(len(img_array)):
            out.write(img_array[i])
        out.release()

# Initialize PPO algorithm
"""CHECK HERE"""
params_sim = {
    "fps":100, #fps
    "step_epsiode":20, #20 seconds
    "n_layer":4, #number of layers
    "n_hidden":128, #number of hidden size.
    "bool_freeze":True, #Train only the last layer.
    "soft_penalty":False,
    "lr_type":"cosine",#"const":constant, "cosine":Cosine scheduler, "mab" : Multi-Armed Bandit algorithm
    "lr_init":3.0e-4,#3e-4
    "interval_eval":10, #evaluation every 1 epicosde.
    "num_eval_episodes":3, #number of episodes per evaluation
    "num_step_train":2.0*10**6, #number of steps for training
    "model_type":"normal", #model type, "with_rope" : with rope's middle points' estimator. "normal"
    "model":"sac", #False, #True : SAC, False:PPO
    "root_dir":r"C:\Users\kawaw\python\pybullet\ur5-bullet\UR5\code\robot_RL_curriculum\residual_policy",
}

#learning rate scheduler
rootDir = params_sim["root_dir"]
bool_soft_penalty=params_sim["soft_penalty"] # add a soft penalty. suppress the model's parameters deviate from the baseline model.

#environment FPS.
_fps = params_sim["fps"] #200 fps
_replay_speed = 1.0
_replay_speed = int(_replay_speed*_fps/20)#replay_speed [frames/s] = _fps/(10frame).rendering is every 10 frames.
_rolloout_length=_fps*params_sim["step_epsiode"] #20 seconds
_step_epsiode = _fps*params_sim["step_epsiode"]
#learning framework
type_data = params_sim["model"]
bool_sac = True if type_data == "sac" else False

type_reward = "mix"
bool_train = True
NUM_EPOCH = int(params_sim["num_step_train"])#1e6: 250 spisodes. 500 episodes.4*10**6
EPOCH = int(NUM_EPOCH/_rolloout_length)
_length_warmup=int(0.1*EPOCH)#int(0.1*EPOCH/_rolloout_length)

lr_type = params_sim["lr_type"]#"const":constant, "cosine":Cosine scheduler, "mab" : Multi-Armed Bandit algorithm
lr_init = params_sim["lr_init"]#3e-4
scheduler = None
if lr_type=="cosine":
    scheduler = lr_scheduler.CosineScheduler(epochs=EPOCH, lr=lr_init, warmup_length=_length_warmup)
#model type
model_type = params_sim["model_type"]
## file setting
postfix = type_data + f"_lrType_{lr_type}"+f"_{params_sim['n_layer']}Layer_{params_sim['n_hidden']}Hidden" + f"_softPenalty{bool_soft_penalty}" + f"_{get_day_time()}"
saveDir = os.path.join(rootDir, postfix)
os.makedirs(saveDir, exist_ok=True)
video_folder = os.path.join(saveDir,"video")
file_base = params_sim["model"] + "_actor_critic.pt"
"""End : tocheck"""

# Initialize environments
id_train = p.connect(p.DIRECT)
id_eval = p.connect(p.DIRECT)
id_render = p.connect(p.DIRECT)
env = UR5eRopeEnvWrapper(fps=_fps,step_episode = _rolloout_length,client_id=id_train)
env_test = UR5eRopeEnvWrapper(fps=_fps,step_episode = _rolloout_length,client_id=id_eval)
env_render = UR5eRopeEnvWrapper(fps=_fps,step_episode = _rolloout_length,client_id=id_render)

#Actor-Critic + PPO
if not bool_sac:
    print("PPO")
    algo = sac.PPO(
        state_shape=env.observation_space.shape,
        action_shape=env.action_space.shape,
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        seed=0,
        n_layer=params_sim["n_layer"],
        n_hidden=params_sim["n_hidden"],
        batch_size=256,
        gamma=0.995,
        lr_actor=lr_init,
        lr_critic=lr_init,#3e-4
        rollout_length=_rolloout_length,
        num_updates=10,
        clip_eps=0.2,
        lambd=0.97,
        coef_ent=0.0,
        max_grad_norm=0.5,
        lr_scheduler = scheduler, 
        lr_type=lr_type,
        model_type=model_type,
        file_pt = file_base,
        bool_soft_penalty=bool_soft_penalty,
        bool_freeze=params_sim["bool_freeze"]
    )
else: #SAC
    print("SAC")
    algo = sac.SAC_alpha(
        state_shape=env.observation_space.shape,
        action_shape=env.action_space.shape,
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        seed=0,
        n_layer=params_sim["n_layer"],
        n_hidden=params_sim["n_hidden"],
        batch_size=256,
        gamma=0.99, 
        lr_actor=lr_init, 
        lr_critic=lr_init,
        lr_alpha=lr_init,
        replay_size=10**6, 
        start_steps=_rolloout_length, 
        tau=5e-3, 
        reward_scale=1.0,
        lr_scheduler = scheduler, 
        lr_type=lr_type,
        model_type=model_type,
        file_pt = file_base,
        bool_soft_penalty=bool_soft_penalty,
        bool_freeze=params_sim["bool_freeze"]
    )
"""CHECK HERE"""

# Load the pretrained weights
#pretrained_weights = torch.load(r'C:\Users\kawaw\python\pybullet\ur5-bullet\UR5\10Hz\ppo_actor_critic.pt')
# # Load the pretrained weights into the actor and critic
# Load the pretrained weights into the actor and critic
#algo.actor.load_state_dict(pretrained_weights['actor_state_dict'])
#algo.critic.load_state_dict(pretrained_weights['critic_state_dict'])

# # If optimizer states are also saved and you want to load them:
# algo.optim_actor.load_state_dict(pretrained_weights['optim_actor'])
# algo.optim_critic.load_state_dict(pretrained_weights['optim_critic'])

# Initialize Trainer
trainer_instance = trainer.Trainer(
    env=env,
    env_test=env_test,
    env_render=env_render,
    algo=algo,
    bool_sac=bool_sac,
    fps=_fps,
    seed=0,
    num_steps=NUM_EPOCH,          # Total training steps
    eval_interval=int(params_sim["interval_eval"]*_rolloout_length),      # Evaluation every 10,000 steps
    num_eval_episodes=params_sim["num_eval_episodes"],       # Number of episodes per evaluation
    step_epsiode=_step_epsiode,
    rootDir=rootDir,
    type_reward=type_data,
    lr_type=lr_type,
    saveDir = saveDir
)

# Start training
trainer_instance.train()
print("Finish training")

if bool_sac:#plot alpha transition
    trainer_instance.plot_alpha_transition()

#plot lr transition.
trainer_instance.plot_lr()

# Plot training progress
trainer_instance.plot()

# Optionally, visualize the trained policy
trainer_instance.visualize()

#make a video
video_path = os.path.join(saveDir, "result.mp4")
imgsize = (640,480)
mkVideo_left = MakeVideo(fps=_replay_speed,imgsize=imgsize,src=video_folder,videoname=video_path)