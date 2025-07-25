# 必要なライブラリのインポート．
from abc import ABC, abstractmethod
import os
import glob
from collections import deque
from time import time
from datetime import timedelta
from base64 import b64encode
import math
import numpy as np
from torch.distributions import Normal
import torch.nn.functional as F
import torch
import torch.nn as nn
import numpy as np
import gym
import matplotlib.pyplot as plt

rootDir = r"C:\Users\kawaw\python\pybullet\ur5-bullet\UR5\code\robot_RL_curriculum\residual_policy"
bool_residual_lr = False #Redidual learning or not.
file_pt = os.path.join(rootDir,"ppo_actor_critic.pt") #Path to the baseline model. 

class MAB:
    """Multi-Armed Bandit Algorithm. 
    Dynamic Learning Rate for Deep Reinforcement Learning: A Bandit Approach
    link : https://arxiv.org/abs/2410.12598
    """
    def __init__(self):
        self._lr_list = [3.75e-5, 7.5e-5, 1.5e-4, 3.0e-4, 6.0e-4, 1.2e-3]  #1e-5~1e-2 
        self._alpha = 0.2 #extracted from the original paper
        self._gamma = 0.98 #discount factor.
        self._n_stack = len(self._lr_list) #number of data to average the previous reward.
        self._fn_list = []  # reward function list.
        self._probability_array = np.zeros(len(self._lr_list)) #probability list for each arm.
        self._weight_list = np.zeros(len(self._lr_list)) #weight list.
        #cumulativer reward
        self._cumulative_reward = 0.0
        #counter
        self._counter = 0
        #previous arm's index.
        self._idx_arm = 3
    
    def _increment(self,reward):
        #add a reward
        self._cumulative_reward+=reward

        #increment a counter.
        self._counter+=1

    def _update(self):
        """
        Update the probability distributino and choose the learing rate.

        Returns:
        --------
        lr_return : float
            learning rate from multi-armed list.
        """
        #averaged reweard.
        f_current = self._cumulative_reward/self._counter
        #save reward to the list.
        self._fn_list.append(f_current)

        #Remove the redundant data.
        if (len(self._fn_list)>self._n_stack):#the volume is larger than the number of stack lis.
            self._fn_list = self._fn_list[1:]

        #update weight
        #Calculate improvement in performance.
        #previous average performance.
        n_back = min(len(self._fn_list),self._n_stack)#data to refer to.
        f_previous = 0
        for i in range(n_back):
            f_previous += self._fn_list[-1*i]
        f_previous /= n_back #average

        #calculate the improvement in performance.
        f_improve = f_current - f_previous

        #update weight.
        for i in range(len(self._weight_list)):#update each arm's weight
            if i==self._idx_arm:#currently chosen arm
                w_current = self._weight_list[i]
                self._weight_list[i] = self._gamma*w_current + self._alpha*f_improve/np.exp(w_current)
            else:#others
                w_current = self._weight_list[i]
                self._weight_list[i] = self._gamma*w_current
        
        #calculate the probability weight.
        weights = self._weight_list #If modify the list, we should use .copy()
        max_weight = np.max(weights)  # for numerical stability
        exp_weights = np.exp(weights - max_weight)
        p_total = np.sum(exp_weights)
        softmax_probs = exp_weights / p_total

        p_stack = 0.0
        for i in range(self._probability_array.shape[0]):
            p_stack += softmax_probs[i]
            self._probability_array[i] = p_stack if (i < self._probability_array.shape[0] - 1) else 1.0
        
        #randomly sample the arm according to self._probability_array
        p_sample = np.random.random()
        idx_list = np.where(self._probability_array>=p_sample)[0]
        #extract the appropriate arm
        idx=len(self._lr_list)-1
        if len(idx_list)>0:
            idx=idx_list[0] #idx_list is guaranteed to be sorted in ascending order.
        
        #update learning rate.
        lr_return = self._lr_list[idx]#next learning rate
        #update arm's index.
        self._idx_arm = idx
        
        #reset
        self._cumulative_reward = 0
        self._counter = 0

        return lr_return


class Algorithm(ABC):

    def explore(self, state, use_base=False):
        """ 確率論的な行動を返す """
        actor = self.actor_base if use_base else self.actor

        state = torch.tensor(state, dtype=torch.float, device=self.device).unsqueeze_(0)
        with torch.no_grad():
            action, log_pi = actor.sample(state)
            state_middle = actor.estimate_middle_point(state)
        return action.cpu().numpy()[0], log_pi.item(), state_middle.cpu().numpy()

    def exploit(self, state, use_base=False):
        """ 決定論的な行動を返す """
        actor = self.actor_base if use_base else self.actor

        state = torch.tensor(state, dtype=torch.float, device=self.device).unsqueeze_(0)
        with torch.no_grad():
            action = actor(state)
        return action.cpu().numpy()[0]

    @abstractmethod
    def is_update(self, steps):
        """ 現在のトータルのステップ数(steps)を受け取り，アルゴリズムを学習するか否かを返す． """
        pass

    @abstractmethod
    def step(self, env, state, t, steps):
        """ 環境(env)，現在の状態(state)，現在のエピソードのステップ数(t)，今までのトータルのステップ数(steps)を
            受け取り，リプレイバッファへの保存などの処理を行い，状態・エピソードのステップ数を更新する．
        """
        pass

    @abstractmethod
    def update(self):
        """ 1回分の学習を行う． """
        pass

def calculate_log_pi(log_stds, noises, actions):
    """ 確率論的な行動の確率密度を返す． """
    # ガウス分布 `N(0, stds * I)` における `noises * stds` の確率密度の対数(= \log \pi(u|a))を計算する．
    # (torch.distributions.Normalを使うと無駄な計算が生じるので，下記では直接計算しています．)
    gaussian_log_probs = \
        (-0.5 * noises.pow(2) - log_stds).sum(dim=-1, keepdim=True) - 0.5 * math.log(2 * math.pi) * log_stds.size(-1)

    # tanh による確率密度の変化を修正する．
    log_pis = gaussian_log_probs - torch.log(1 - actions.pow(2) + 1e-6).sum(dim=-1, keepdim=True)

    return log_pis

def reparameterize(means, log_stds):
    """ Reparameterization Trickを用いて，確率論的な行動とその確率密度を返す． """
    # 標準偏差．
    stds = log_stds.exp()
    # 標準ガウス分布から，ノイズをサンプリングする．
    noises = torch.randn_like(means)
    # Reparameterization Trickを用いて，N(means, stds)からのサンプルを計算する．
    us = means + noises * stds
    # tanh　を適用し，確率論的な行動を計算する．
    actions = torch.tanh(us)

    # 確率論的な行動の確率密度の対数を計算する．
    log_pis = calculate_log_pi(log_stds, noises, actions)

    return actions, log_pis

def atanh(x):
    """ tanh の逆関数． """
    return 0.5 * (torch.log(1 + x + 1e-6) - torch.log(1 - x + 1e-6))


def evaluate_lop_pi(means, log_stds, actions):
    """ 平均(mean)，標準偏差の対数(log_stds)でパラメータ化した方策における，行動(actions)の確率密度の対数を計算する． """
    noises = (atanh(actions) - means) / (log_stds.exp() + 1e-8)
    return calculate_log_pi(log_stds, noises, actions)

"""PPO + Actor-Critic"""
class PPOActor(nn.Module):

    def __init__(self, state_shape, action_shape,n_layer=2,n_hidden=128):
        super().__init__()

        layers = [nn.Linear(state_shape[0], n_hidden), nn.Tanh()]
        for _ in range(n_layer - 1):
            layers.append(nn.Linear(n_hidden, n_hidden))
            layers.append(nn.Tanh())
        layers.append(nn.Linear(n_hidden, action_shape[0]))
        self.net = nn.Sequential(*layers)

        self.log_stds = nn.Parameter(torch.zeros(1, action_shape[0]))

    def forward(self, states):
        return self.net(states) #unbounded output#torch.tanh(self.net(states))

    def sample(self, states):
        return reparameterize(self.net(states), self.log_stds)

    def evaluate_log_pi(self, states, actions):
        return evaluate_lop_pi(self.net(states), self.log_stds, actions)

    def estimate_middle_point(self, states):
        #return dummy ouputs
        batch_size = states.shape[0]
        device = states.device
        out = torch.zeros(batch_size, 7, device=device)
        return out.squeeze(0)
    
class PPOActorWithEstimator(nn.Module):
    def __init__(self, state_shape, action_shape,n_layer=2,n_hidden=128):
        super().__init__()

        self.shared = [nn.Linear(state_shape[0], n_hidden), nn.Tanh()]
        for _ in range(n_layer - 1):
            self.shared.append(nn.Linear(n_hidden, n_hidden))
            self.shared.append(nn.Tanh())


        # Action branch
        self.action_head = nn.Linear(n_hidden, action_shape[0])
        self.log_stds = nn.Parameter(torch.zeros(1, action_shape[0]))

        # Middle point estimator branch
        self.middle_estimator = nn.Sequential(
            nn.Linear(256, 7),
            nn.Tanh(),
            nn.Linear(7, 7)  # length+position + velocity size
        )

    def forward(self, states):
        features = self.shared(states)
        actions = torch.tanh(self.action_head(features))  # bounded output
        return actions

    def sample(self, states):
        features = self.shared(states)
        mus = self.action_head(features)
        actions = reparameterize(mus, self.log_stds)
        return actions

    def evaluate_log_pi(self, states, actions):
        features = self.shared(states)
        mus = self.action_head(features)
        return evaluate_lop_pi(mus, self.log_stds, actions)

    def estimate_middle_point(self, states):
        features = self.shared(states)
        state = self.middle_estimator(features)
        return state.squeeze(0) # squeeze the first dimension if it is 1.

class PPOCritic(nn.Module):
    def __init__(self, state_shape, n_layer=2, n_hidden=128):
        super().__init__()

        layers = [nn.Linear(state_shape[0], n_hidden), nn.Tanh()]
        for _ in range(n_layer - 1):
            layers.append(nn.Linear(n_hidden, n_hidden))
            layers.append(nn.Tanh())
        layers.append(nn.Linear(n_hidden, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, states):
        return self.net(states)
       

def calculate_advantage(values, rewards, dones, next_values, gamma=0.995, lambd=0.997):
    """ GAEを用いて，状態価値のターゲットとGAEを計算する． """

    # TD誤差を計算する．
    deltas = rewards + gamma * next_values * (1 - dones) - values

    # GAEを初期化する．
    advantages = torch.empty_like(rewards)

    # 終端ステップを計算する．
    advantages[-1] = deltas[-1]

    # 終端ステップの1つ前から，順番にGAEを計算していく．
    for t in reversed(range(rewards.size(0) - 1)):
        advantages[t] = deltas[t] + gamma * lambd * (1 - dones[t]) * advantages[t + 1]

    # 状態価値のターゲットをλ-収益として計算する．
    targets = advantages + values

    # GAEを標準化する．
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    return targets, advantages

class RolloutBuffer:

    def __init__(self, buffer_size, state_shape, action_shape, device=torch.device('cuda')):

        # GPU上に保存するデータ．
        self.states = torch.empty((buffer_size, *state_shape), dtype=torch.float, device=device)
        self.actions = torch.empty((buffer_size, *action_shape), dtype=torch.float, device=device)
        self.rewards = torch.empty((buffer_size, 1), dtype=torch.float, device=device)
        self.rewards_base = torch.empty((buffer_size, 1), dtype=torch.float, device=device)
        self.dones = torch.empty((buffer_size, 1), dtype=torch.float, device=device)
        self.log_pis = torch.empty((buffer_size, 1), dtype=torch.float, device=device)
        self.next_states = torch.empty((buffer_size, *state_shape), dtype=torch.float, device=device)
        self.state_rope_middle_estimate = torch.empty((buffer_size, 7), dtype=torch.float, device=device)
        self.state_rope_middle = torch.empty((buffer_size, 7), dtype=torch.float, device=device)

        # 次にデータを挿入するインデックス．
        self._p = 0
        self._p_base = 0
        self.buffer_size = buffer_size

    def append(self, state, action, reward, done, log_pi, next_state,state_rope_middle,state_rope_middle_estimate):
        self.states[self._p].copy_(torch.from_numpy(state))
        self.actions[self._p].copy_(torch.from_numpy(action))
        self.rewards[self._p] = float(reward)
        #self.rewards_base[self._p] = float(reward_base)
        self.dones[self._p] = float(done)
        self.log_pis[self._p] = float(log_pi)
        self.next_states[self._p].copy_(torch.from_numpy(next_state))
        self.state_rope_middle[self._p].copy_(torch.from_numpy(state_rope_middle))
        self.state_rope_middle_estimate[self._p].copy_(torch.from_numpy(state_rope_middle_estimate))
        self._p = (self._p + 1) % self.buffer_size
        #print(f"{self._p=}")

    def append_base(self,reward_base):
        self.rewards_base[self._p_base] = float(reward_base)
        self._p_base = (self._p_base + 1) % self.buffer_size
        #print(f"{self._p_base=}")

    def get(self):
        #assert self._p == 0, 'Buffer needs to be full before training.'
        print(f"Get data from buffer. {self._p=}, {self._p_base=}")
        self._p = 0
        self._p_base = 0
        return self.states, self.actions, self.rewards,self.rewards_base, self.dones, self.log_pis, self.next_states,self.state_rope_middle,self.state_rope_middle_estimate
     

class PPO(Algorithm):

    def __init__(self, state_shape, action_shape, device=torch.device('cuda'), seed=0,
                 n_layer=2,n_hidden=128, batch_size=64, gamma=0.995, lr_actor=3e-4, lr_critic=3e-4,
                 rollout_length=2048, num_updates=10, clip_eps=0.2, lambd=0.97,
                 coef_ent=0.0, max_grad_norm=0.5,lr_scheduler = None, 
                 lr_type="const",model_type="with_rope",file_pt = file_pt,
                 bool_soft_penalty=False,bool_freeze=True#add a soft penalyt
                 ):
        super().__init__()

        # シードを設定する．
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

        self.bool_soft_penalty=bool_soft_penalty
        self.lambd_l2 = 1e-6 #1e-5 : almost no constraint. 1e-4:middle constraint (reasonable). 1e-3 : strong constraint.
        self._r_max_penalty = 0.02 #up to 2% for the total loss.

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # データ保存用のバッファ．
        self.buffer = RolloutBuffer(
            buffer_size=rollout_length,
            state_shape=state_shape,
            action_shape=action_shape,
            device=device
        )

        #Initialize the Actor-Critic model
        self.actor = PPOActor(
            state_shape=state_shape,
            action_shape=action_shape,
            n_layer=n_layer,
            n_hidden=n_hidden
        ).to(device)
        #baseline model
        self.actor_base= PPOActor(
            state_shape=state_shape,
            action_shape=action_shape,
            n_layer=n_layer,
            n_hidden=n_hidden
        ).to(device)

        self.critic = PPOCritic(
            state_shape=state_shape,
            n_layer=n_layer,
            n_hidden=n_hidden
        ).to(device)
        #baseline model
        self.critic_base = PPOCritic(
            state_shape=state_shape,
            n_layer=n_layer,
            n_hidden=n_hidden
        ).to(device)
        ####

        #initialize optimizer
        self.optim_actor = torch.optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.optim_critic = torch.optim.Adam(self.critic.parameters(), lr=lr_critic)
        ###

        # Load pretrained weights if available
        if os.path.exists(file_pt):
            print(f"Loading pretrained model from {file_pt}")
            checkpoint = torch.load(file_pt, map_location=device)
            #model
            self.actor.load_state_dict(checkpoint['actor'])
            #self.critic.load_state_dict(checkpoint['critic'])
            #baseline model.
            self.actor_base.load_state_dict(checkpoint['actor'])
            self.critic_base.load_state_dict(checkpoint['critic'])
            #optimizer
            #self.optim_actor.load_state_dict(checkpoint['optim_actor'])
            #self.optim_critic.load_state_dict(checkpoint['optim_critic'])
            
            # Freeze all layers except the last layer
            if bool_freeze:
                self._freeze_layers_except_last()
            self.print_trainable_parameters()  # Debug: show which parameters are trainable
        else:
            print(f"No pretrained model found at {file_pt}. Training from scratch.")

        #####################

        #Learning rate scheduler with MAB algorithm
        self.lr_scheduler = lr_scheduler
        self.lr_type = lr_type

        self.lr_scheduler_mab_actor = MAB()
        self.lr_scheduler_mab_critic = MAB()
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic

        # その他パラメータ．
        self.learning_steps = 0
        self.device = device
        self.batch_size = batch_size
        self.gamma = gamma
        self.rollout_length = rollout_length
        self.num_updates = num_updates
        self.clip_eps = clip_eps
        self.lambd = lambd
        self.coef_ent = coef_ent
        self.max_grad_norm = max_grad_norm

    def is_update(self, steps):
        # ロールアウト1回分のデータが溜まったら学習する．
        return steps % self.rollout_length == 0

    def step(self, env, state, t, steps):
        t += 1

        # Exploration actor (stochastic, for actual env action) 
        if not env.bool_base:#False : for training
            action, log_pi, hat_state_middle = self.explore(state, use_base=False)#trained model's action
            # Interact with environment using exploration action
            next_state, state_rope_middle, reward, done, _ = env.step(action,bool_base=False)
            
            if t >= env.max_episode_steps:
                done_masked = False
            else:
                done_masked = done

            # バッファにデータを追加する．
            self.buffer.append(state, action, reward, done_masked, log_pi, next_state,state_rope_middle,hat_state_middle)
        else:#baseline
            # Exploitation actor (deterministic, for monitoring)
            action_base = self.exploit(state, use_base=True)
            #baseline model's result
            next_state, _, reward_base, done, _ = env.step(action_base,bool_base=True)
            
            if t >= env.max_episode_steps:
                done_masked = False
            else:
                done_masked = done

            # バッファにデータを追加する．
            self.buffer.append_base(reward_base)


        # エピソードが終了した場合には，環境をリセットする．
        if done:
            t = 0
            #next_state = env.reset()

        return next_state, t,done

    def update(self,num_step):
        """
         obs = np.concatenate([
            end_effector_position, #3
            end_effector_orientation, #3 (6)
            vel_eef, #6 (12)
            self.pos_human_current, #3 (15)
            self.vel_human, #3 (18)
            net_force, #3 (21)
            net_torque, #3 (24)
            state_rope_middle  #7. (length,pos,vel) (31)
        ]).astype(np.float32)
        """

        self.learning_steps += 1
        #learning rate scheduler
        if self.lr_type=="cosine":#cosine scheduler
            #print(f"{num_step=},{self.rollout_length=}")
            epoch = int(num_step/self.rollout_length)
            new_lr = self.lr_scheduler(epoch) #update learning rate
            #Actor
            for param_group in self.optim_actor.param_groups:
                param_group['lr'] = new_lr
            #Critic
            for param_group in self.optim_critic.param_groups:
                param_group['lr'] = new_lr

        states, actions, rewards,rewards_base, dones, log_pis, next_states,state_rope_middle,hat_state_middle = self.buffer.get()

        #critic part
        with torch.no_grad():
            values = self.critic(states)
            next_values = self.critic(next_states)

        #Actor part
        #use reward difference.
        rewards_diff = rewards - rewards_base
        #use normal reward.
        #targets, advantages = calculate_advantage(values, rewards, dones, next_values, self.gamma, self.lambd)
        #use reward difference.
        targets, advantages = calculate_advantage(values, rewards_diff, dones, next_values, self.gamma, self.lambd)

        # バッファ内のデータを num_updates回ずつ使って，ネットワークを更新する．
        for _ in range(self.num_updates):#for each step
            # インデックスをシャッフルする．
            indices = np.arange(self.rollout_length)
            np.random.shuffle(indices)

            # ミニバッチに分けて学習する．
            for start in range(0, self.rollout_length, self.batch_size): #mini-batch learning to consider uncertainty.
                idxes = indices[start:start+self.batch_size]
                self.update_critic(states[idxes], targets[idxes])
                self.update_actor(states[idxes], actions[idxes], log_pis[idxes], advantages[idxes],state_rope_middle[idxes],hat_state_middle[idxes])
            
        #update learning rate.
        if self.lr_type=="mab":#Multi-armed bandit
            #lr for actor
            self.lr_actor = self.lr_scheduler_mab_actor._update()
            #update learning rate.
            for param_group in self.optim_actor.param_groups:
                param_group['lr'] = self.lr_actor
            
            #lr for critic
            self.lr_critic = self.lr_scheduler_mab_critic._update()
            #update learning rate
            for param_group in self.optim_critic.param_groups:
                param_group['lr'] = self.lr_critic

    def update_critic(self, states, targets):
        loss_critic = (self.critic(states) - targets).pow_(2).mean()
        
        #increment the state of lr_scheduler. negative loss is adopted to enlarge the reward.
        if self.lr_type=="mab":
            reward_critic = -1.0*loss_critic.item() #transform to python float to torch.tensor.
            self.lr_scheduler_mab_critic._increment(reward=reward_critic)

        self.optim_critic.zero_grad()
        loss_critic.backward(retain_graph=False)
        # 学習を安定させるヒューリスティックとして，勾配のノルムをクリッピングする．
        nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
        self.optim_critic.step()

    def update_actor(self, states, actions, log_pis_old, advantages,state_rope_middle,hat_state_middle):
        log_pis = self.actor.evaluate_log_pi(states, actions)
        mean_entropy = -log_pis.mean()

        ratios = (log_pis - log_pis_old).exp_()

        loss_actor1 = -ratios * advantages

        loss_actor2 = -torch.clamp(
            ratios,
            1.0 - self.clip_eps,
            1.0 + self.clip_eps
        ) * advantages

        loss_actor = torch.max(loss_actor1, loss_actor2).mean() - self.coef_ent * mean_entropy

        loss_middle = F.mse_loss(hat_state_middle, state_rope_middle)

        #soft penalty
        l2_loss = 0.0
        for param_actor, param_actor_base in zip(self.actor.parameters(), self.actor_base.parameters()):
            l2_loss += torch.nn.functional.mse_loss(param_actor, param_actor_base, reduction='sum')
        #print(f"{l2_loss=},{loss_actor=}")
        #add a soft penalty.
        if self.bool_soft_penalty:
            loss_actor_scalar = abs(loss_actor.item())
            loss_actor += min(loss_actor_scalar*self._r_max_penalty,self.lambd_l2*l2_loss)

        # Combine if using shared optimizer
        total_loss = loss_actor + loss_middle
        

        #increment the state of lr_scheduler. negative loss is adopted to enlarge the reward.
        if self.lr_type=="mab":#MAB
            reward_actor = -1.0*loss_actor.item()
            self.lr_scheduler_mab_actor._increment(reward=reward_actor)

        self.optim_actor.zero_grad()
        loss_actor.backward(retain_graph=False)

        # 学習を安定させるヒューリスティックとして，勾配のノルムをクリッピングする．
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        self.optim_actor.step()
""""""

"""SAC + Automatic adjustment of alpha"""
class SACActor(nn.Module):

    def __init__(self, state_shape, action_shape, n_layer=2, n_hidden=128):
        super().__init__()

        # Build layers dynamically
        layers = [nn.Linear(state_shape[0], n_hidden), nn.ReLU(inplace=True)]
        for _ in range(n_layer - 1):
            layers.append(nn.Linear(n_hidden, n_hidden))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(n_hidden, 2 * action_shape[0]))  # mu and sigma
        self.net = nn.Sequential(*layers)

    def forward(self, states):
        return torch.tanh(self.net(states).chunk(2, dim=-1)[0])

    def sample(self, states):
        # [演習] 確率論的な行動と確率密度の対数を計算し，返します．
        #split network output into mean and std.
        means, log_stds = self.net(states).chunk(2, dim=-1)
        #clip data between -20 and 2.
        log_stds = log_stds.clamp(-20, 2)
        return reparameterize(means,log_stds)

    def estimate_middle_point(self, states):
        #return dummy ouputs
        batch_size = states.shape[0]
        device = states.device
        out= torch.zeros(batch_size, 7, device=device)
        return out.squeeze(0)

class SACActorWithEstimator(nn.Module):
    def __init__(self, state_shape, action_shape, n_layer=2, n_hidden=128):
        super().__init__()

        # Build shared layers dynamically
        shared_layers = [nn.Linear(state_shape[0], n_hidden), nn.ReLU(inplace=True)]
        for _ in range(n_layer - 1):
            shared_layers.append(nn.Linear(n_hidden, n_hidden))
            shared_layers.append(nn.ReLU(inplace=True))
        self.shared = nn.Sequential(*shared_layers)

        # Action branch (outputs mean and log_std)
        self.action_head = nn.Linear(n_hidden, 2 * action_shape[0])  # mean and log_std

        # Middle point estimator branch (position + velocity)
        self.middle_estimator = nn.Sequential(
            nn.Linear(n_hidden, 7),
            nn.ReLU(inplace=True),
            nn.Linear(7, 7)
        )

    def forward(self, states):
        features = self.shared(states)
        means, _ = self.action_head(features).chunk(2, dim=-1)
        return torch.tanh(means)

    def sample(self, states):
        features = self.shared(states)
        means, log_stds = self.action_head(features).chunk(2, dim=-1)
        log_stds = log_stds.clamp(-20, 2)
        return reparameterize(means, log_stds)

    def evaluate_log_pi(self, states, actions):
        features = self.shared(states)
        means, log_stds = self.action_head(features).chunk(2, dim=-1)
        log_stds = log_stds.clamp(-20, 2)
        return evaluate_lop_pi(means, log_stds, actions)

    def estimate_middle_point(self, states):
        features = self.shared(states)
        out = self.middle_estimator(features)
        return out.squeeze(0)
    
class SACCritic(nn.Module):

    def __init__(self, state_shape, action_shape, n_layer=2, n_hidden=128):
        super().__init__()

        # Build first critic network dynamically
        net1_layers = [nn.Linear(state_shape[0] + action_shape[0], n_hidden), nn.ReLU(inplace=True)]
        for _ in range(n_layer - 1):
            net1_layers.append(nn.Linear(n_hidden, n_hidden))
            net1_layers.append(nn.ReLU(inplace=True))
        net1_layers.append(nn.Linear(n_hidden, 1))
        self.net1 = nn.Sequential(*net1_layers)
        
        # Build second critic network dynamically
        net2_layers = [nn.Linear(state_shape[0] + action_shape[0], n_hidden), nn.ReLU(inplace=True)]
        for _ in range(n_layer - 1):
            net2_layers.append(nn.Linear(n_hidden, n_hidden))
            net2_layers.append(nn.ReLU(inplace=True))
        net2_layers.append(nn.Linear(n_hidden, 1))
        self.net2 = nn.Sequential(*net2_layers)

    def forward(self, states, actions):
        # [演習] ソフト状態行動価値を2つ計算し，返します．
        #concatenate states and actions along the last dimension.
        x = torch.cat([states,actions],dim=-1)
        return self.net1(x),self.net2(x)


class ReplayBuffer:

    def __init__(self, buffer_size, state_shape, action_shape, device):
        # 次にデータを挿入するインデックス．
        self._p = 0
        self._p_base = 0
        # データ数．
        self._n = 0
        # リプレイバッファのサイズ．
        self.buffer_size = buffer_size

        # GPU上に保存するデータ．
        self.states = torch.empty((buffer_size, *state_shape), dtype=torch.float, device=device)
        self.actions = torch.empty((buffer_size, *action_shape), dtype=torch.float, device=device)
        self.rewards = torch.empty((buffer_size, 1), dtype=torch.float, device=device)
        self.rewards_base = torch.empty((buffer_size, 1), dtype=torch.float, device=device)
        self.dones = torch.empty((buffer_size, 1), dtype=torch.float, device=device)
        self.next_states = torch.empty((buffer_size, *state_shape), dtype=torch.float, device=device)
        self.state_rope_middle = torch.empty((buffer_size, 7), dtype=torch.float, device=device)#(x,y,z,vx,vy,vz)
        self.state_rope_middle_estimate = torch.empty((buffer_size, 7), dtype=torch.float, device=device)#(x,y,z,vx,vy,vz)

    def append(self, state, action, reward, done, next_state, state_rope_middle, state_rope_middle_estimate):
        self.states[self._p].copy_(torch.from_numpy(state))
        self.actions[self._p].copy_(torch.from_numpy(action))
        self.rewards[self._p] = float(reward)
        self.dones[self._p] = float(done)
        self.next_states[self._p].copy_(torch.from_numpy(next_state))
        self.state_rope_middle[self._p].copy_(torch.from_numpy(state_rope_middle))
        self.state_rope_middle_estimate[self._p].copy_(torch.from_numpy(state_rope_middle_estimate))

        self._p = (self._p + 1) % self.buffer_size
        self._n = min(self._n + 1, self.buffer_size)

    def append_base(self, reward_base):
        self.rewards_base[self._p_base] = float(reward_base)
        self._p_base = (self._p_base + 1) % self.buffer_size

    def sample(self, batch_size):
        idxes = np.random.randint(low=0, high=self._n, size=batch_size)
        return (
            self.states[idxes],
            self.actions[idxes],
            self.rewards[idxes],
            self.rewards_base[idxes],
            self.dones[idxes],
            self.next_states[idxes],
            self.state_rope_middle[idxes],
            self.state_rope_middle_estimate[idxes]
        )
    
class SAC_alpha(Algorithm):

    def __init__(self, state_shape, action_shape, device=torch.device('cuda' if torch.cuda.is_available() else "cpu"), seed=0,
                 n_layer=2,n_hidden=128,batch_size=256, gamma=0.99, lr_actor=3e-4, lr_critic=3e-4,lr_alpha=3e-4,
                 replay_size=10**6, start_steps=10**4, tau=5e-3, reward_scale=1.0,
                 lr_scheduler = None, lr_type="const",model_type="normal"
                 ,file_pt = file_pt,bool_soft_penalty=False,bool_freeze=True):
        super().__init__()

        # シードを設定する．
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

        # リプレイバッファ．
        self.buffer = ReplayBuffer(
            buffer_size=replay_size,
            state_shape=state_shape,
            action_shape=action_shape,
            device=device,
        )

        # Actor-Criticのネットワークを構築する．
        if model_type!="with_rope": #normal
            self.actor = SACActor(
                state_shape=state_shape,
                action_shape=action_shape,
                n_layer=n_layer,
                n_hidden=n_hidden
            ).to(device)
            self.actor_base = SACActor(
                state_shape=state_shape,
                action_shape=action_shape,
                n_layer=n_layer,
                n_hidden=n_hidden
            ).to(device)
        else:#with middle points' estimator
            self.actor = SACActorWithEstimator(
                state_shape=state_shape,
                action_shape=action_shape,
                n_layer=n_layer,
                n_hidden=n_hidden
            ).to(device)
            self.actor_base = SACActorWithEstimator(
                state_shape=state_shape,
                action_shape=action_shape,
                n_layer=n_layer,
                n_hidden=n_hidden
            ).to(device)

        self.critic = SACCritic(
            state_shape=state_shape,
            action_shape=action_shape,
            n_layer=n_layer,
            n_hidden=n_hidden
        ).to(device)
        self.critic_base = SACCritic(
            state_shape=state_shape,
            action_shape=action_shape,
            n_layer=n_layer,
            n_hidden=n_hidden
        ).to(device)

        self.critic_target = SACCritic(
            state_shape=state_shape,
            action_shape=action_shape,
            n_layer=n_layer,
            n_hidden=n_hidden
        ).to(device).eval()
        self.critic_target_base = SACCritic(
            state_shape=state_shape,
            action_shape=action_shape,
            n_layer=n_layer,
            n_hidden=n_hidden
        ).to(device).eval()

        #laod pretrained model.
        if os.path.exists(file_pt):
            print(f"Loading pretrained model from {file_pt}")
            checkpoint = torch.load(file_pt, map_location=device)
            #model
            self.actor.load_state_dict(checkpoint['actor'])
            #self.critic.load_state_dict(checkpoint['critic'])
            #self.critic_target.load_state_dict(checkpoint['critic'])
            #baseline model.
            self.actor_base.load_state_dict(checkpoint['actor'])
            self.critic_base.load_state_dict(checkpoint['critic'])
            self.critic_target_base.load_state_dict(checkpoint['critic'])

            # Freeze all layers except the last layer in actor
            if bool_freeze:
                self._freeze_layers_except_last()
            self.print_trainable_parameters()  # Debug: show which parameters are trainable
        
        else:
            print(f"No pretrained model found at {file_pt}. Training from scratch.")


        # Initialize the target networks weight.
        self.critic_target.load_state_dict(self.critic.state_dict())
        for param in self.critic_target.parameters():
            param.requires_grad = False
        for param in self.critic_target_base.parameters():
            param.requires_grad = False

        #Learning rate scheduler with MAB algorithm
        self.lr_type=lr_type
        self.lr_scheduler = lr_scheduler
        #MAB
        self._counter_update_lr = 10
        self.lr_scheduler_mab_actor = MAB()
        self.lr_scheduler_mab_critic = MAB()
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic

        # オプティマイザ．
        self.optim_actor = torch.optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.optim_critic = torch.optim.Adam(self.critic.parameters(), lr=lr_critic)

        # その他パラメータ．
        self.learning_steps = 0
        self.batch_size = batch_size
        self.device = device
        self.gamma = gamma
        self.start_steps = start_steps
        self.tau = tau
        self.reward_scale = reward_scale

        #Initializatin for alpha
        self.target_entropy = -np.prod(action_shape)  # heuristic value
        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        self.alpha = self.log_alpha.exp()*0.2
        self.optim_alpha = torch.optim.Adam([self.log_alpha], lr=lr_alpha)

        self.alpha_values = []

    def is_update(self, steps):
        # 学習初期の一定期間(start_steps)は学習しない．
        return steps >= max(self.start_steps, self.batch_size)

    def step(self, env, state, t, steps):
        t += 1

        # Divide behavior for residual (training) and baseline (monitoring) modes for PPO residual policy
        if not env.bool_base:  # Training with residual policy
            if steps <= self.start_steps:
                action = env.action_space.sample()
                log_pi = 0.0
                state_middle_hat = np.zeros(7)
            else:
                action, log_pi, state_middle_hat = self.explore(state, use_base=False)
            next_state, state_rope_middle, reward, done, _ = env.step(action, bool_base=False)

            if t == env.max_episode_steps:
                done_masked = False
            else:
                done_masked = done

            # Store residual policy experience in buffer
            self.buffer.append(state, action, reward, done_masked, next_state, state_rope_middle, state_middle_hat)
        else:  # Baseline policy (for monitoring)
            action_base = self.exploit(state, use_base=True)
            next_state, _, reward_base, done, _ = env.step(action_base, bool_base=True)

            if t == env.max_episode_steps:
                done_masked = False
            else:
                done_masked = done

            # Store only baseline reward in buffer
            self.buffer.append_base(reward_base)
        if done:
            t = 0
            #next_state = env.reset()

        return next_state, t,done

    def update(self,num_step):
        self.learning_steps += 1
        if self.lr_type=="cosine":#cosine scheduler
            epoch=int(num_step/self.start_steps)
            new_lr = self.lr_scheduler(epoch) #update learning rate
            #Actor
            for param_group in self.optim_actor.param_groups:
                param_group['lr'] = new_lr
            #Critic
            for param_group in self.optim_critic.param_groups:
                param_group['lr'] = new_lr

        #Mini-batch sampling.
        states, actions, rewards,rewards_base, dones, next_states,state_rope_middle,state_rope_middle_estimate = self.buffer.sample(self.batch_size)

        #residual reward
        rewards_residual = rewards - rewards_base

        #adopt the residual reward.
        self.update_critic(states, actions, rewards_residual, dones, next_states)
        self.update_actor_and_alpha(states,state_rope_middle,state_rope_middle_estimate)
        self.update_target()

        #MAB : update learning rate.
        if self.lr_type=="mab":
            if self.lr_scheduler_mab_actor._counter%self._counter_update_lr==0:
                #lr for actor
                self.lr_actor = self.lr_scheduler_mab_actor._update()
                #update learning rate.
                for param_group in self.optim_actor.param_groups:
                    param_group['lr'] = self.lr_actor
                
                #lr for critic
                self.lr_critic = self.lr_scheduler_mab_critic._update()
                #update learning rate
                for param_group in self.optim_critic.param_groups:
                    param_group['lr'] = self.lr_critic

    def update_critic(self, states, actions, rewards, dones, next_states):
        curr_qs1, curr_qs2 = self.critic(states, actions)

        with torch.no_grad():
            next_actions, log_pis = self.actor.sample(next_states)
            next_qs1, next_qs2 = self.critic_target(next_states, next_actions)
            next_qs = torch.min(next_qs1, next_qs2) - self.alpha * log_pis
        target_qs = rewards * self.reward_scale + (1.0 - dones) * self.gamma * next_qs

        loss_critic1 = (curr_qs1 - target_qs).pow_(2).mean()
        loss_critic2 = (curr_qs2 - target_qs).pow_(2).mean()

        self.optim_critic.zero_grad()
        (loss_critic1 + loss_critic2).backward(retain_graph=False)

        if self.lr_type=="mab":
            #increment the state of lr_scheduler. negative loss is adopted to enlarge the reward.
            reward_critic = -1.0*(loss_critic1.item()+loss_critic2.item()) #transform to python float to torch.tensor.
            self.lr_scheduler_mab_critic._increment(reward=reward_critic)
        
        self.optim_critic.step()

    def update_actor_and_alpha(self, states,state_rope_middle,state_rope_middle_estimate):
        actions, log_pis = self.actor.sample(states)
        qs1, qs2 = self.critic(states, actions)
        #adopt tuned alpha
        loss_actor = (self.alpha.detach() * log_pis - torch.min(qs1, qs2)).mean()

        self.optim_actor.zero_grad()
        loss_actor.backward()
        self.optim_actor.step()

        #increment the state of lr_scheduler. negative loss is adopted to enlarge the reward.
        if self.lr_type=="mab":
            reward_actor = -1.0*loss_actor.item()
            self.lr_scheduler_mab_actor._increment(reward=reward_actor)

        # update Alpha loss
        alpha_loss = -(self.log_alpha * (log_pis + self.target_entropy).detach()).mean()

        self.optim_alpha.zero_grad()
        alpha_loss.backward()
        self.optim_alpha.step()

        self.alpha = self.log_alpha.exp()
        self.alpha_values.append(self.alpha.item())

    def update_target(self):
        for t, s in zip(self.critic_target.parameters(), self.critic.parameters()):
            t.data.mul_(1.0 - self.tau)
            t.data.add_(self.tau * s.data)

    def _freeze_layers_except_last(self):
        """
        Freeze actor hidden layers (keep last layer trainable) but keep critic fully trainable.
        This allows the critic to adapt to new reward functions while preserving actor features.
        """
        # Handle different actor network structures
        if hasattr(self.actor, 'net'):  # PPOActor or SACActor
            # Freeze actor hidden layers, keep last layer trainable
            for i, layer in enumerate(self.actor.net):
                if isinstance(layer, nn.Linear):
                    if i < len(self.actor.net) - 2:  # All layers except the last linear layer
                        layer.weight.requires_grad = False
                        layer.bias.requires_grad = False
                    else:  # Last linear layer - keep trainable
                        layer.weight.requires_grad = True
                        layer.bias.requires_grad = True
        elif hasattr(self.actor, 'shared'):  # PPOActorWithEstimator or SACActorWithEstimator
            # Freeze shared layers
            for layer in self.actor.shared:
                if isinstance(layer, nn.Linear):
                    layer.weight.requires_grad = False
                    layer.bias.requires_grad = False
            
            # Keep action head trainable
            self.actor.action_head.weight.requires_grad = True
            self.actor.action_head.bias.requires_grad = True
        
        # Keep critic fully trainable (no freezing)
        # The critic needs to adapt to the new reward function
        for param in self.critic.parameters():
            param.requires_grad = True
        
        # Freeze base models completely (they should remain frozen)
        for param in self.actor_base.parameters():
            param.requires_grad = False
        for param in self.critic_base.parameters():
            param.requires_grad = False
            
        print("Frozen actor hidden layers (last layer trainable). Critic is fully trainable.")
        
    def print_trainable_parameters(self):
        """
        Print which parameters are trainable for debugging.
        """
        print("\n=== Trainable Parameters ===")
        print("Actor trainable parameters:")
        for name, param in self.actor.named_parameters():
            if param.requires_grad:
                print(f"  {name}: {param.shape}")
        
        print("\nCritic trainable parameters:")
        for name, param in self.critic.named_parameters():
            if param.requires_grad:
                print(f"  {name}: {param.shape}")
        
        print("\nBase models (should be frozen):")
        actor_base_trainable = sum(p.requires_grad for p in self.actor_base.parameters())
        critic_base_trainable = sum(p.requires_grad for p in self.critic_base.parameters())
        print(f"  Actor base trainable params: {actor_base_trainable}")
        print(f"  Critic base trainable params: {critic_base_trainable}")
        print("===========================\n") 