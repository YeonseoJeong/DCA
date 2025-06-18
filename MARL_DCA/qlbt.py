import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
import time
import sys
import os
sys.path.append(os.path.dirname(__file__))  # 현재 MARL_DCA 폴더


import gymnasium as gym
import wandb
from pettingzoo.mpe import simple_spread_v3
from pettingzoo.utils.conversions import aec_to_parallel
from utils.logger import Logger
from utils.RnnReplaybuffer import ReplayBufferRNN

'''
QMIX
Lqmix: E(Sigma i=1^bs)[(yi_tot - Qi_tot(τ, u, s; θ))^2] 
yi_tot = r +γ maxu′ Qtot(τ ′, u′, s′; θ−)
Qtot(τ, u, s; θ) = f(Q1(τ1, u1, s; θ), Q2(τ2, u2, s; θ), ..., Qn(τn, un, s; θ); s)
f: mixing network
'''

# 추가한 부분
# QMIX - agent net에 Q_indiv를 추가하여, 각 에이전트의 Q값을 개별적으로 계산하고 joint 해서 Mixing Network에 전달

class AgentNetwork(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim=64):
        super(AgentNetwork, self).__init__()
        self.hidden_dim = hidden_dim
        self.fc1 = nn.Linear(obs_dim + action_dim, hidden_dim)
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)  # 히스토리 저장
        self.q_out = nn.Linear(hidden_dim, action_dim)

    def forward(self, obs, last_action, his_in):
        x = torch.cat([obs, last_action], dim=-1)
        x = F.relu(self.fc1(x))
        if his_in is None:
            his_in = torch.zeros(x.size(0), self.hidden_dim, device=x.device)
        else:
            try:
                his_in = his_in.reshape(x.size(0), self.hidden_dim)
            except Exception as e:
                raise ValueError(f"Invalid hidden state shape: {his_in.shape}") from e

        his_out = self.gru(x, his_in) 
        q_ind = self.q_out(his_out)
        return q_ind, his_out


class HyperNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(HyperNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU()
        )
    def forward(self, state):
        return torch.abs(self.fc(state))


class MixingNetwork(nn.Module):
    def __init__(self, n_agents, state_dim, hidden_dim=64):
        super(MixingNetwork, self).__init__()
        self.n_agents = n_agents
        self.hidden_dim = hidden_dim
        self.state_dim = state_dim

        self.hyper_w1 = HyperNetwork(state_dim, n_agents * hidden_dim)
        self.hyper_b1 = nn.Linear(state_dim, hidden_dim)

        # For joint Q_indiv (vectorized)
        self.hyper_w2_ind = HyperNetwork(state_dim, hidden_dim * n_agents)
        self.hyper_b2_ind = nn.Linear(state_dim, n_agents)

        # For Q_total (scalar)
        self.hyper_w2_tot = HyperNetwork(state_dim, hidden_dim)
        self.hyper_b2_tot = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, q_indiv, state): 
        # q_indiv : (batch, n_agents)
        # state : (batch, state_dim) = obs_dim * n_agents

        q_indiv = q_indiv.unsqueeze(0) if q_indiv.dim() == 1 else q_indiv       # [B, n_agents]
        state = state.unsqueeze(0) if state.dim() == 1 else state               # [B, state_dim]

        # First Layer
        w1 = self.hyper_w1(state).view(-1, self.n_agents, self.hidden_dim)      # W1 = [B, n_agents, hidden_dim]
        b1 = self.hyper_b1(state).view(-1, 1, self.hidden_dim)                  # b1 = [B, 1, hidden_dim]
        hidden = F.elu(torch.bmm(q_indiv.unsqueeze(1), w1) + b1)                # hidden = [B, 1, hidden_dim]        

        # ---- Q_joint_indiv (vector output) ----
        w2_ind = self.hyper_w2_ind(state).view(-1, self.hidden_dim, self.n_agents)      # W2_ind = [B, hidden_dim, n_agents]
        b2_ind = self.hyper_b2_ind(state).view(-1, 1, self.n_agents)                    # b2_ind = [B, 1, n_agents]
        joint_q_ind = torch.bmm(hidden, w2_ind) + b2_ind                        # joint_q_ind = [B, 1, n_agents]
        joint_q_ind = joint_q_ind.squeeze(1)                                    # joint_q_ind = [B, n_agents]
        
        # ---- Q_total (scalar output) ----
        w2_tot = self.hyper_w2_tot(state).view(-1, self.hidden_dim, 1)              # W2_tot = [B, hidden_dim, 1]
        b2_tot = self.hyper_b2_tot(state)                                           # b2_tot = [B, 1]    
        q_total = torch.bmm(hidden, w2_tot).squeeze(1) + b2_tot                     # q_total = [B, 1]

        return joint_q_ind, q_total


class QLBT_Agent(nn.Module):
    def __init__(self,
                env,
                hidden_dims, 
                batch_size = 64, 
                buffer_capacity = 10000, 
                lr = 0.0003,
                gamma=0.95, 
                epochs = 10,
                max_steps = 200,
                log_dir = "logs/qlbt_logs",
                plot_window = 100,
                clip_grad = 10.0,
                update_interval=100, 
                device = "cpu",
                tau = 0.01,
                decay_ratio = 0.99
                ):
        super(QLBT_Agent, self).__init__()
        
        # Environment
        self.env = env
        self.env.reset()
        self.agents = env.agents
        self.n_agents = len(self.agents) # N
        self.device = torch.device(device)
        self.buffer = ReplayBufferRNN(capacity=buffer_capacity, device =self.device)
        self.hidden_dims = hidden_dims
        self.log_prefix = "qlbt_"

        self.agent_nets = nn.ModuleDict()
        self.target_agent_nets = nn.ModuleDict()
        self.obs_spaces = {}

        # Initialize agent networks and target networks
        for agent in self.agents:
            # obs_space = env.observation_space[agent]
            obs_space = env.observation_space(agent)
            
            if isinstance(obs_space, gym.spaces.Dict):
                obs_dim = sum(space.n if isinstance(space, gym.spaces.Discrete) else space.shape[0] for space in obs_space.spaces.values())
            else:
                obs_dim = obs_space.n if isinstance(obs_space, gym.spaces.Discrete) else obs_space.shape[0]
            
            # act_dim = self.env.action_space[agent].n
            self.act_dim = self.env.action_space(agent).n
            self.agent_nets[agent] = AgentNetwork(obs_dim, self.act_dim, hidden_dims).to(self.device)
            self.target_agent_nets[agent] = AgentNetwork(obs_dim, self.act_dim, hidden_dims).to(self.device)
            self.obs_spaces[agent] = obs_space
        
        ''' agent_nets의 네트워크 파라미터도 optimizer에 포함시켜야 함'''
        agent_params = []
        for agent in self.agent_nets.values():
            agent_params += list(agent.parameters())

        self.mixing_net = MixingNetwork(self.n_agents, obs_dim * self.n_agents, hidden_dims).to(self.device)
        self.target_mixing_net = MixingNetwork(self.n_agents, obs_dim * self.n_agents, hidden_dims).to(self.device)
        self.optimizer = optim.Adam(agent_params+list(self.mixing_net.parameters()), lr=lr,amsgrad=True)

        self.batch_size = batch_size # B
        self.gamma = gamma
        self.update_interval = update_interval
        self.epochs = epochs
        self.max_steps = max_steps
        self.clip_grad = clip_grad
        self.epsilon_start = 1.0
        self.epsilon_end = 0.1
        self.decay_ratio = decay_ratio
        self.step = 0
        self.tau = tau
        self.logger = Logger(log_dir, self.log_prefix, plot_window)
        self.update_target(tau=None)

    def preprocess_observation(self, obs, agent):
        # Convert Dictionary observation to a flat tensor
        obs_space = self.obs_spaces[agent]

        if isinstance(obs_space, gym.spaces.Dict):
            one_hots = []
            for key in obs_space.spaces.keys():
                value = obs[key]
                if isinstance(obs_space.spaces[key], gym.spaces.Discrete):
                    n = obs_space.spaces[key].n
                    one_hot = torch.zeros(n, device = self.device)
                    one_hot[int(value)] = 1.0
                    one_hots.append(one_hot)
                else:
                    v = np.array(value, dtype=np.float32).flatten()
                    one_hots.append(torch.from_numpy(v).to(self.device))
            obs_tensor = torch.cat(one_hots)
        elif isinstance(obs, np.ndarray):
            obs_tensor = torch.FloatTensor(obs).to(self.device)
        else:
            raise TypeError(f"Unsupported observation type: {type(obs)}")
        return obs_tensor.unsqueeze(0)
    
    ### soft update
    def update_target(self, tau=None):  # tau=None이면 hard update
        if tau is None:
            for agent in self.agents:
                self.target_agent_nets[agent].load_state_dict(self.agent_nets[agent].state_dict())
            self.target_mixing_net.load_state_dict(self.mixing_net.state_dict())
        else:
            for agent in self.agents:
                for target_param, param in zip(self.target_agent_nets[agent].parameters(), self.agent_nets[agent].parameters()):
                    target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)
            for target_param, param in zip(self.target_mixing_net.parameters(), self.mixing_net.parameters()):
                target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

    '''너무 단순하게 감소하는 경향이 있음'''
    def epsilon_decay(self):
        self.epsilon_start = max(self.epsilon_end, self.epsilon_start * self.decay_ratio)
        return self.epsilon_start 
    

    def td_target(self, rewards, target_qs, done_mask, gamma):
        if rewards.dim() > 2:
            rewards = rewards.squeeze(-1)
        if target_qs.dim() > 2:
            target_qs = target_qs.squeeze(-1)
        if done_mask.dim() > 2:
            done_mask = done_mask.squeeze(-1)

        B, T = rewards.shape
        targets = torch.zeros_like(rewards).to(rewards.device)
        targets[:, -1] = rewards[:, -1]  # Last Q-value
        
        for t in reversed(range(T - 1)):
            targets[:, t] = rewards[:, t] + gamma * target_qs[:, t + 1] * (1.0 - done_mask[:, t + 1])
        return targets


    def select_action(self, agent, obs, last_action, h_in):

        obs_tensor = torch.FloatTensor(obs).to(self.device)
        obs_tensor = obs_tensor.unsqueeze(0) if obs_tensor.dim() == 1 else obs_tensor
        
        last_action_tensor = torch.FloatTensor(last_action).to(self.device) 
        last_action_tensor = last_action_tensor.unsqueeze(0) if last_action_tensor.dim() == 1 else last_action_tensor
            
        q_values, h_out = self.agent_nets[agent](obs_tensor, last_action_tensor, h_in)
        q_values = q_values.squeeze(0) if q_values.dim() > 1 and q_values.size(0) == 1 else q_values
            
        if random.random() < self.epsilon_start: # exploration
            action = random.randint(0, self.act_dim - 1)
        else: # exploitation
            action = q_values.argmax().item()
       
        return action, h_out


    def rollout_episode(self):
        env = self.env
        obs_dict, _ = env.reset()
        episode_data = []

        obs = {agent: self.preprocess_observation(obs_dict[agent], agent) for agent in self.agents}
        last_actions = {
            agent: torch.zeros(1, self.act_dim, device=self.device)
            for agent in self.agents
        }
        hidden_states = {
            agent: torch.zeros(1, self.agent_nets[agent].hidden_dim, device=self.device)
            for agent in self.agents
        }
        
        for _ in range(self.max_steps):
            actions = {}
            for agent in self.agents:
                action, hidden_states[agent] = self.select_action(agent, obs[agent], last_actions[agent], hidden_states[agent])
                actions[agent] = action

            next_obs, rewards, terminations, truncations, infos = env.step(actions)
            # env.render()

            next_obs_proc = {
                agent: self.preprocess_observation(next_obs[agent], agent) for agent in self.agents
            }
            joint_obs = torch.stack([obs[agent].squeeze(0) if obs[agent].dim() == 2 else obs[agent] for agent in self.agents])
            joint_next_obs = torch.stack([next_obs_proc[agent].squeeze(0) if next_obs_proc[agent].dim() == 2 else next_obs_proc[agent] for agent in self.agents])
            joint_actions = torch.tensor([actions[agent] for agent in self.agents])
            joint_rewards = torch.tensor([rewards[agent] for agent in self.agents])
            joint_dones = torch.tensor([terminations[agent] or truncations[agent] for agent in self.agents])

            episode_data.append((joint_obs, joint_actions, joint_rewards, joint_next_obs, joint_dones))
            obs = next_obs_proc
            last_actions = {
                agent: F.one_hot(torch.tensor(actions[agent]), num_classes=self.act_dim).float().to(self.device)
                for agent in self.agents
            }

            if all(terminations.values()) or all(truncations.values()):
                break

        s_seq, a_seq, r_seq, ns_seq, d_seq = zip(*episode_data)
        self.buffer.push(
            state_seq=torch.stack(s_seq),
            action_seq=torch.stack(a_seq),
            reward_seq=torch.stack(r_seq),
            next_state_seq=torch.stack(ns_seq),
            done_seq=torch.stack(d_seq)
        )

    def update(self, alpha=0.2, seq_len=10):
        if len(self.buffer) < self.batch_size:
            return 0.0

        state, action, reward, next_state, done = self.buffer.sample(self.batch_size, seq_len)
        B, T, N, obs_dim = state.shape
        agent_qs, target_qs = [], []

        for i, agent in enumerate(self.agents):
            action_i = action[:, :, i]                      # action_i = (B, T)
            obs_i = state[:, :, i, :]                       # obs_i = (B, T, obs)
            nextobs_i = next_state[:, :, i, :]              # nextobs_i = (B, T, obs)

            h, h_target = None, None
            q_seq, target_q_seq = [], []

            for t in range(T):                              # T: sequence length
                obs_t = obs_i[:, t]                         # obs_t = (B, obs)
                act_t = action_i[:, t]                      # act_t = (B,)
                a_onehot_t = F.one_hot(act_t, num_classes=self.act_dim).float()  # (B, A)

                q, h = self.agent_nets[agent](obs_t, a_onehot_t, h)         # (B, ), (B, hidden_dim)
                q_selected = q.gather(-1, act_t.unsqueeze(-1)).squeeze(-1)  # (B, T)
                q_seq.append(q_selected.unsqueeze(1))  # (B, 1)

                with torch.no_grad():
                    next_obs_t = nextobs_i[:, t]  # (B, obs_dim)
                    q_target_all, h_target = self.target_agent_nets[agent](next_obs_t, a_onehot_t, h_target)  # (B, A), (B, hidden_dim)
                    next_a = q_target_all.argmax(dim=-1, keepdim=True)         # (B, T, 1)
                    q_target = q_target_all.gather(-1, next_a).squeeze(-1)  # (B, T)
                    target_q_seq.append(q_target.unsqueeze(1))  # (B, 1)

            agent_qs.append(torch.cat(q_seq, dim =1))           
            target_qs.append(torch.cat(target_q_seq, dim =1))   

        agent_qs = torch.stack(agent_qs, dim=-1)     # (B, T, N)
        target_qs = torch.stack(target_qs, dim=-1)   # (B, T, N)

        state = state.view(B, T, -1)
        next_state = next_state.view(B, T, -1)

        # Mixing Network
        joint_q_ind_list, q_total_list = [], []
        target_q_ind_list, tq_total_list = [], []
        for t in range(T):
            joint_q_t, q_total_t = self.mixing_net(agent_qs[:, t], state[:, t])
            joint_q_ind_list.append(joint_q_t.unsqueeze(1))  # (B, 1, N)
            q_total_list.append(q_total_t.unsqueeze(1))          # (B, 1)

            with torch.no_grad():
                tq_ind_t, tq_total_t = self.target_mixing_net(target_qs[:, t], next_state[:, t])
                target_q_ind_list.append(tq_ind_t.unsqueeze(1))
                tq_total_list.append(tq_total_t.unsqueeze(1))
        
        joint_q_ind = torch.cat(joint_q_ind_list, dim=1)  # (B, T, N)
        q_total = torch.cat(q_total_list, dim=1).squeeze(-1)           # (B, T)
        target_joint_q_ind = torch.cat(target_q_ind_list, dim=1)  # (B, T, N)
        tq_total = torch.cat(tq_total_list, dim=1).squeeze(-1)            # (B, T)


        r_total = reward.sum(dim=2)                     # (B, T)
        done_mask = done.any(dim=2).float()             # (B, T)
        y_total = self.td_target(r_total, tq_total, done_mask, self.gamma)

        # Loss
        loss_qmix = F.mse_loss(q_total, y_total.detach())
        loss_ind = F.mse_loss(joint_q_ind, target_joint_q_ind.detach())
        loss = loss_qmix + alpha * loss_ind
        self.optimizer.zero_grad()
        loss.backward()
        if self.clip_grad is not None:
            nn.utils.clip_grad_norm_(self.parameters(), max_norm=self.clip_grad)
        self.optimizer.step()

        self.epsilon_decay()
        self.step += 1
        self.update_target(tau=self.tau)

        per_agent_qs = agent_qs.detach().mean(dim=(0, 1))
        avg_q_total = q_total.detach().mean().item()
        td_errors = (agent_qs - target_qs.detach()) ** 2
        per_agent_losses = td_errors.mean(dim=(0, 1))

        metrics = {
            'avg_loss': loss.item(),
            'avg_reward': r_total.mean().item(),
            'avg_q_total': avg_q_total,
            'avg_entropy': self.epsilon_decay(),
            'per_agent_qs': {agent: per_agent_qs[i].item() for i, agent in enumerate(self.agents)},
            'per_agent_losses': {agent: per_agent_losses[i].item() for i, agent in enumerate(self.agents)}
        }
        return metrics
    

    def train(self, log_interval=1):
        wandb.init(
            project="QLBT",      
            name=f"run_{int(time.time())}",  
            config={
                "alpha": 0.2,
                "seq_len": 10,
                "epochs": self.epochs,
                "batch_size": self.batch_size,
            }
        )
        for episode in range(self.epochs):
            self.rollout_episode()
            metrics = self.update(alpha=0.2, seq_len=10) 
            if not metrics:
                continue

            # Log overall metrics
            log_data = {
                'avg_reward': metrics['avg_reward'],
                'avg_q_total': metrics['avg_q_total'],
                'avg_loss': metrics['avg_loss'],
                'avg_entropy': metrics['avg_entropy'],
            }
            # Log per-agent metrics
            for agent in self.agents:
                log_data[f'{agent}_q_indiv'] = metrics['per_agent_qs'][agent]
                log_data[f'{agent}_loss'] = metrics['per_agent_losses'][agent]
            
            # self.logger.log_metrics(log_data, episode)
            wandb.log(log_data, step=episode)
            try:
                self.env.save_render_data(save_dir="logs/render_data", episode=episode)
            except Exception as e:
                print(f"[ERROR] save_render_data failed at episode {episode}: {e}")

        #     if episode % log_interval == 0:
        #         self.logger.info(f"Episode {episode} | Avg Reward: {metrics['avg_reward']:.4f} | Avg Q total: {metrics['avg_q_total']:.4f} | Avg Loss: {metrics['avg_loss']:.4f} | Avg Entropy: {metrics['avg_entropy']:.4f}") 
        # self.logger.close()       


    def save(self, path):
        checkpoint = {
        "agent_nets": {agent: net.state_dict() for agent, net in self.agent_nets.items()},
        "mixing_net": self.mixing_net.state_dict(),
        "optimizer": self.optimizer.state_dict(),
        "step": self.step,  # 선택적: 현재 학습 단계
        "args": {
            "hidden_dims": self.hidden_dims,
            "gamma": self.gamma,
            "batch_size": self.batch_size,
            # 필요한 하이퍼파라미터들 추가
            }
        }
        torch.save(checkpoint, path)
        print(f"[SAVE] Model saved to {path}")

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)

        for agent in self.agent_nets:
            self.agent_nets[agent].load_state_dict(checkpoint["agent_nets"][agent])
            
        self.mixing_net.load_state_dict(checkpoint["mixing_net"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.training_steps = checkpoint.get("step", 0)

        print(f"[LOAD] Model loaded from {path}")


if __name__ == '__main__':
    env = simple_spread_v3.parallel_env(render_mode = 'None', N=3, max_cycles = 200, continuous_actions=False)
    # env = aec_to_parallel(env)
    hidden_dims = 128

    qmix = QLBT_Agent(env=env, hidden_dims=hidden_dims, batch_size=64, buffer_capacity=10000, lr=0.0003, gamma=0.95,
                epochs=1000, max_steps=500, log_dir="logs/simple_spread_logs", plot_window=100,
                update_interval=100, device="cpu", tau=0.01)

    qmix.train()
    
