import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
from collections import deque

import gymnasium as gym
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

# 1. hidden_seq -> buffer에 넣을 필요 x
# 2. seq단위로 history를 저장하고, agent별로 hidden state를 유지

class AgentNetwork(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim=64):
        super(AgentNetwork, self).__init__()

        self.hidden_dim = hidden_dim
        self.fc1 = nn.Linear(obs_dim + action_dim, hidden_dim)
        self.gru = nn.GRUCell(hidden_dim, hidden_dim) 
        self.q_out = nn.Linear(hidden_dim, action_dim)
        self.hidden_state = None
    
    def reset_hidden(self, batch_size=1, device=None):
        device = device or next(self.parameters()).device
        self.hidden_state = torch.zeros(batch_size, self.hidden_dim, device=device)

    def forward(self, obs, last_action):
        x = torch.cat([obs, last_action], dim=-1)
        x = F.relu(self.fc1(x))

        if self.hidden_state is None:
            self.reset_hidden(batch_size = x.size(0), device=x.device)
        
        self.hidden_state = self.hidden_state.view(x.size(0), self.hidden_dim)  # Ensure his_in is the right shape
        self.hidden_state = self.gru(x, self.hidden_state) 
        q = self.q_out(self.hidden_state)
        return q


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

        self.hyper_w2 = HyperNetwork(state_dim, hidden_dim)
        self.hyper_b2 = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1) # Q_tot
        )

    def forward(self, agents_q, state): 
        # agents_q : [q1, q2, ... , qn]
        # state : [state_dim] = obs_dim * n_agents

        agents_q = agents_q.unsqueeze(0) if agents_q.dim() == 1 else agents_q   # [1, n_agents]
        state = state.unsqueeze(0) if state.dim() ==1 else state                # [1, state_dim]

        w1 = self.hyper_w1(state).view(-1, self.n_agents, self.hidden_dim)  # W1 = [1, n_agents, hidden_dim]
        b1 = self.hyper_b1(state).view(-1, 1, self.hidden_dim)              # b1 = [1, 1, hidden_dim]
        
        hidden = F.elu(torch.bmm(agents_q.unsqueeze(1), w1) + b1).squeeze(1)             
        # hidden = [1, 1, n_agents] * [1, n_agents, hidden_dim] + [1, 1, hidden_dim]
        # hidden = [1, 1, hidden_dim(뒤에 2, 3차원이 곱해짐 앞에 batch_제외)] + [1, 1, hidden_dim] -> [1, hidden_dim]

        w2 = self.hyper_w2(state).view(-1, self.hidden_dim, 1)              # W2 = [1, hidden_dim, 1]
        b2 = self.hyper_b2(state)                                           # b2 = [1, 1] -> [1, 1]

        q_total = torch.bmm(hidden.unsqueeze(1), w2).squeeze(1) + b2
        # q_total = [1, 1, hidden_dim] * [1, hidden_dim, 1] + [1, 1]
        # q_total = [1, 1] + [1, 1] -> [1, 1]
        return q_total.squeeze()      # [1]

    

def td_lambda_target(rewards, target_qs, gamma=0.95, td_lambda=0.8):
    if rewards.dim() > 2:
        rewards = rewards.squeeze(-1)
    if target_qs.dim() > 2:
        target_qs = target_qs.squeeze(-1)

    B, T = rewards.shape
    targets = torch.zeros_like(rewards).to(rewards.device)
    targets[:, -1] = target_qs[:, -1].detach()  # Last target Q value
    for t in reversed(range(T - 1)):
        targets[:, t] = rewards[:, t] + gamma * (
            td_lambda * targets[:, t + 1] + (1 - td_lambda) * target_qs[:, t + 1]
        ) # td_lambda * target_qs[:, t + 1] + (1 - td_lambda) * targets[:, t + 1]
    return targets



class QMIX(nn.Module):
    def __init__(self,
                env,
                hidden_dims, 
                batch_size = 64, 
                buffer_capacity = 10000, 
                lr=0.0003, 
                gamma=0.95, 
                epochs = 10,
                max_steps = 200,
                log_dir = "logs/qmix_dca_logs",
                plot_window = 100,
                clip_grad = None,
                update_interval=100, 
                device="cpu",
                tau=None,
                decay_ratio = 0.98
                ):
        super(QMIX, self).__init__()
        
        # Environment
        self.env = env
        self.env.reset()
        if hasattr(env, "aec_env"):
            for landmark in env.aec_env.unwrapped.world.landmarks:
                landmark.state.p_vel = np.zeros(2)
                landmark.movable = False
                landmark.collide = False
        else:
            pass
        self.agents = env.agents
        self.n_agents = len(self.agents) # N
        self.device = torch.device(device)
        self.buffer = ReplayBufferRNN(buffer_capacity, device =self.device)

        self.log_prefix = "qmix_" + "dca"


        self.agent_nets = nn.ModuleDict()
        self.target_agent_nets = nn.ModuleDict()
        self.obs_spaces = {}

        for agent in self.agents:
            obs_space = env.observation_space[agent]
            
            if isinstance(obs_space, gym.spaces.Dict):
                obs_dim = sum(space.n if isinstance(space, gym.spaces.Discrete) else space.shape[0] for space in obs_space.spaces.values())
            else:
                obs_dim = obs_space.n if isinstance(obs_space, gym.spaces.Discrete) else obs_space.shape[0]
            
            act_dim = self.env.action_space[agent].n

            self.agent_nets[agent] = AgentNetwork(obs_dim, act_dim, hidden_dims).to(self.device)
            self.target_agent_nets[agent] = AgentNetwork(obs_dim, act_dim, hidden_dims).to(self.device)
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
        self.max_steps = max_steps # T -> history 저장 길이..
        self.clip_grad = clip_grad
        self.epsilon_start = 1.0
        self.epsilon_end = 0.05
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
    
    ### 3. soft update
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


    def epsilon_decay(self):
        self.epsilon_start = max(self.epsilon_end, self.epsilon_start * self.decay_ratio)  #1, 0.995, 0.990, 0.985 ... 0.01
        return self.epsilon_start 


    def select_action(self, agent, obs, last_action):

        obs_tensor = torch.FloatTensor(obs).to(self.device) # (1, obs_dim)
        if obs_tensor.dim() == 1:
            obs_tensor = obs_tensor.unsqueeze(0)
        
        last_action_tensor = torch.FloatTensor(last_action).to(self.device) 
        if last_action_tensor.dim() == 1:
            last_action_tensor = last_action_tensor.unsqueeze(0)
        
        q_values = self.agent_nets[agent](obs_tensor, last_action_tensor)
        q_values = q_values.squeeze(0) if q_values.dim() > 1 and q_values.size(0) == 1 else q_values
            
        if random.random() < self.epsilon_start:
            action = random.randint(0, self.env.action_space[agent].n - 1)
        else:
            action = q_values.argmax().item()

       
        return action

        
    def update(self, alpha=0.1, seq_len=10):
        if len(self.buffer) < self.batch_size:
            return 0.0

        state, action, reward, next_state, done = self.buffer.sample(self.batch_size, seq_len)
        B, T, N, obs_dim = state.shape
        # print(f"[DEBUG] state.shape = {state.shape}, action.shape = {action.shape}, reward.shape = {reward.shape}, next_state.shape = {next_state.shape}, done.shape = {done.shape}")
        
        agent_qs, target_qs = [], []

        for i, agent in enumerate(self.agents):
            a_i = action[:, :, i]                    # (B, T)
            s_i = state[:, :, i, :]                  # (B, T, obs)
            ns_i = next_state[:, :, i, :]            # (B, T, obs)

            # batch size = B
            self.agent_nets[agent].reset_hidden(batch_size=B, device=s_i.device)
            self.target_agent_nets[agent].reset_hidden(batch_size=B, device=s_i.device)

            q_seq, target_q_seq = [], []

            for t in range(T):
                obs_t = s_i[:, t]  # (B, obs_dim)
                act_t = a_i[:, t]  # (B,)
                a_onehot_t = F.one_hot(act_t, num_classes=self.env.action_space[agent].n).float()  # (B, A)

                q = self.agent_nets[agent](obs_t, a_onehot_t)
                q_selected = q.gather(-1, act_t.unsqueeze(-1)).squeeze(-1)  # (B, T)
                q_seq.append(q_selected.unsqueeze(1))  # (B, 1)

                with torch.no_grad():
                    next_obs_t = ns_i[:, t]  # (B, obs_dim)
                    q_target_all = self.target_agent_nets[agent](next_obs_t, a_onehot_t)
                    next_a = q_target_all.argmax(dim=-1, keepdim=True)         # (B, T, 1)
                    q_target = q_target_all.gather(-1, next_a).squeeze(-1)  # (B, T)
                    target_q_seq.append(q_target.unsqueeze(1))  # (B, 1)

            agent_qs.append(torch.cat(q_seq, dim =1))       # (B, T)
            target_qs.append(torch.cat(target_q_seq, dim =1)) # (B, T)

        agent_qs = torch.stack(agent_qs, dim=-1)     # (B, T, N)
        target_qs = torch.stack(target_qs, dim=-1)   # (B, T, N)

        state = state.view(B, T, -1)
        next_state = next_state.view(B, T, -1)

        q_total = torch.stack([self.mixing_net(agent_qs[:, t], state[:, t]) for t in range(T)], dim=1)
        tq_total = torch.stack([self.target_mixing_net(target_qs[:, t], next_state[:, t]) for t in range(T)], dim=1)

        r_total = reward.sum(dim=2)                   # (B, T)
        y_total = td_lambda_target(r_total, tq_total, gamma=self.gamma, td_lambda=0.8)

        # Loss
        loss_qmix = F.mse_loss(q_total, y_total.detach())

        loss_ind = 0.0
        for i in range(self.n_agents):
            agent_q = agent_qs[:, :, i]
            target_q = target_qs[:, :, i]
            loss_ind += F.mse_loss(agent_q, target_q.detach())
        loss_ind /= self.n_agents

        loss = loss_qmix + alpha * loss_ind
        self.optimizer.zero_grad()
        loss.backward()
        if self.clip_grad is not None:
            nn.utils.clip_grad_norm_(self.parameters(), max_norm=self.clip_grad)
        self.optimizer.step()

        self.epsilon_decay()
        self.step += 1
        self.update_target()

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
    
    def rollout_episode(self):
        env = self.env
        obs_dict, _ = env.reset()

        if hasattr(env, "aec_env"):
            for landmark in env.aec_env.unwrapped.world.landmarks:
                landmark.state.p_vel = np.zeros(2)
                landmark.movable = False
                landmark.collide = False

        episode_data = []

        obs = {agent: self.preprocess_observation(obs_dict[agent], agent) for agent in self.agents}
        last_actions = {
            agent: torch.zeros(1, self.env.action_space[agent].n, device=self.device)
            for agent in self.agents
        }
        
        for agent in self.agents:
            self.agent_nets[agent].reset_hidden(batch_size=1, device=self.device)
            
        for _ in range(self.max_steps):
            actions = {}
            for agent in self.agents:
                action = self.select_action(agent, obs[agent], last_actions[agent])
                actions[agent] = action

            next_obs, rewards, terminations, truncations, infos = env.step(actions)

            next_obs_proc = {
                agent: self.preprocess_observation(next_obs[agent], agent) for agent in self.agents
            }
            joint_obs = torch.stack([obs[agent].squeeze(0) if obs[agent].dim() == 2 else obs[agent] for agent in self.agents])
            joint_next_obs = torch.stack([next_obs_proc[agent].squeeze(0) if next_obs_proc[agent].dim() == 2 else next_obs_proc[agent] for agent in self.agents])
            joint_actions = torch.tensor([actions[agent] for agent in self.agents])
            joint_rewards = torch.tensor([rewards[agent] for agent in self.agents])
            joint_dones = torch.tensor([terminations[agent] for agent in self.agents])

            episode_data.append((joint_obs, joint_actions, joint_rewards, joint_next_obs, joint_dones))

            obs = next_obs_proc

            last_actions = {
                agent: F.one_hot(torch.tensor(actions[agent]), num_classes=self.env.action_space[agent].n).float().to(self.device)
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




    def train(self, max_episode, log_interval = 1):
        for episode in range(max_episode):
            self.rollout_episode() # returns: {agent_0: [q0, q1, ..., qT], ...}
            metrics = self.update(alpha=0.1, seq_len=10) # returns: {avg_loss, avg_reward, avg_q_total, per_agent_qs, per_agent_losses}

            if not metrics:
                continue

            # Log overall metrics
            log_data = {
                'avg_reward': metrics['avg_reward'],
                'avg_q_total': metrics['avg_q_total'],
                'avg_loss': metrics['avg_loss'],
                #'avg_entropy': metrics['avg_entropy'],
            }

            # Log per-agent metrics
            for agent in self.agents:
                log_data[f'{agent}_q_indiv'] = metrics['per_agent_qs'][agent]
                log_data[f'{agent}_loss'] = metrics['per_agent_losses'][agent]
            
            self.logger.log_metrics(log_data, episode)

            try:
                self.env.save_render_data(save_dir="logs/render_data", episode=episode)
            except Exception as e:
                print(f"[ERROR] save_render_data failed at episode {episode}: {e}")

            if episode % log_interval == 0:
                self.logger.info(f"Episode {episode} | Avg Reward: {metrics['avg_reward']:.4f} | Avg Q total: {metrics['avg_q_total']:.4f} | Avg Loss: {metrics['avg_loss']:.4f} | Avg Entropy: {metrics['avg_entropy']:.4f}") 
        self.logger.close()    
        


    def save(self, path):
        checkpoint = {
        "agent_nets": {agent: net.state_dict() for agent, net in self.agent_nets.items()},
        "mixing_net": self.mixing_net.state_dict(),
        "optimizer": self.optimizer.state_dict(),
        "step": self.training_steps,  # 선택적: 현재 학습 단계
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



# if __name__ == '__main__':
#     env = simple_spread_v3.parallel_env(render_mode = 'None', N=3, max_cycles = 200, continuous_actions=False)
#     if hasattr(env, "aec_env"):
#         for agent in env.aec_env.unwrapped.world.agents:
#             agent.size = 0.02
#     else:
#         pass
#     # env = aec_to_parallel(env)
#     hidden_dims = 128

#     qmix = QMIX(env=env, hidden_dims=hidden_dims, batch_size=64, buffer_capacity=10000, lr=0.0003, gamma=0.95,
#                 epochs=10, max_steps=200, log_dir="logs/qmix_simple_spread_logs", plot_window=100,
#                 update_interval=100, device="cpu", tau=0.01)

#     qmix.train(max_episode=1000, log_interval=1)
    
