import torch
import torch.nn as nn
import torch.nn.functional as F

'''
QMIX
Lqmix: E(Sigma i=1^bs)[(yi_tot - Qi_tot(τ, u, s; θ))^2] 
yi_tot = r +γ maxu′ Qtot(τ ′, u′, s′; θ−)
Qtot(τ, u, s; θ) = f(Q1(τ1, u1, s; θ), Q2(τ2, u2, s; θ), ..., Qn(τn, un, s; θ); s)
f: mixing network
'''

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

        # === Q_total (scalar) ===
        self.hyper_w1 = HyperNetwork(state_dim, n_agents * hidden_dim)
        self.hyper_b1 = nn.Linear(state_dim, hidden_dim)

        self.hyper_w2_tot = HyperNetwork(state_dim, hidden_dim)
        self.hyper_b2_tot = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        # === joint_q_indiv (vectorized) ===
        self.indiv_w2 = nn.Linear(state_dim, hidden_dim * n_agents)
        self.indiv_b2 = nn.Linear(state_dim, n_agents)
        

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
        w2_ind = self.indiv_w2(state).view(-1, self.hidden_dim, self.n_agents)      # W2_ind = [B, hidden_dim, n_agents]
        b2_ind = self.indiv_b2(state)                                       # b2_ind = [B, n_agents]
        joint_q_ind = torch.bmm(hidden, w2_ind).squeeze(1) + b2_ind             # joint_q_ind = [B, n_agents]
        
        # ---- Q_total (scalar output) ----
        w2_tot = self.hyper_w2_tot(state).view(-1, self.hidden_dim, 1)              # W2_tot = [B, hidden_dim, 1]
        b2_tot = self.hyper_b2_tot(state)                                           # b2_tot = [B, 1]    
        q_total = torch.bmm(hidden, w2_tot).squeeze(1) + b2_tot                     # q_total = [B, 1]

        return joint_q_ind, q_total
