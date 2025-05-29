import random
from collections import deque
import torch

class ReplayBufferRNN:
    """
    Replay buffer for RNN-based agents in QMIX (GRU 기반).
    Stores full trajectories per episode, and returns fixed-length sequences during training.
    """
    def __init__(self, capacity, device="cpu"):
        self.buffer = deque(maxlen=capacity)
        self.device = device

    def push(self, state_seq, action_seq, reward_seq, next_state_seq, done_seq):
        '''
        Stores one full episode
        - state_seq: (T, N, obs_dim)
        - action_seq: (T, N)
        - reward_seq: (T, N)
        - next_state_seq: (T, N, obs_dim)
        - done_seq: (T, N)
        '''
        if len(self.buffer) == self.buffer.maxlen:
            self.buffer.popleft()
        data = (
            state_seq.detach(),
            action_seq.detach(),
            reward_seq.detach(),
            next_state_seq.detach(),
            done_seq.detach()
        )
        self.buffer.append(data)

    def sample(self, batch_size, seq_len):
        """
        Samples fixed-length subsequences from stored full episodes.
        Returns:
            - state:      (B, seq_len, N, obs_dim)
            - action:     (B, seq_len, N)
            - reward:     (B, seq_len, N)
            - next_state: (B, seq_len, N, obs_dim)
            - done:       (B, seq_len, N)
        """
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = [], [], [], [], []

        while len(state_batch) < batch_size:
            s_seq, a_seq, r_seq, ns_seq, d_seq = random.choice(self.buffer)
            T = s_seq.size(0)
            if T < seq_len:
                continue
            start_idx = random.randint(0, T - seq_len)
            end_idx = start_idx + seq_len
            state_batch.append(s_seq[start_idx:end_idx])
            action_batch.append(a_seq[start_idx:end_idx])
            reward_batch.append(r_seq[start_idx:end_idx])
            next_state_batch.append(ns_seq[start_idx:end_idx])
            done_batch.append(d_seq[start_idx:end_idx])
            
        s_tensor = torch.stack(state_batch).to(self.device)        # (B, seq_len, N, obs_dim)
        a_tensor = torch.stack(action_batch).to(self.device)       # (B, seq_len, N)
        r_tensor = torch.stack(reward_batch).to(self.device)       # (B, seq_len, N)
        ns_tensor = torch.stack(next_state_batch).to(self.device)  # (B, seq_len, N, obs_dim)
        d_tensor = torch.stack(done_batch).to(self.device)         # (B, seq_len, N)

        return s_tensor, a_tensor, r_tensor, ns_tensor, d_tensor

    def __len__(self):
        return len(self.buffer)
    

class ReplayBufferRNN2: 
    '''
    state = joint_obs,
    joint_action,                
    reward,                        
    next_state = next_joint_obs, 
    joint_hidden_state,          
    joint_done        
    .detach()를 통해 gradient를 끊어, backpropagation을 하지 않음 -> 메모리 누수나 에러 방지            
    '''
    def __init__(self, capacity=10000, device="cpu"):
        self.buffer = deque(maxlen=capacity)
        self.device = device

    def push(self, hidden_seq, state_seq, action_seq, reward_seq, next_state_seq, dones):
        if len(self.buffer) == self.buffer.maxlen:
            self.buffer.popleft()


        data = (hidden_seq.detach(), 
                state_seq.detach(), 
                action_seq.detach(), 
                reward_seq.detach(), 
                next_state_seq.detach(), 
                dones.detach()
        )
        self.buffer.append(data)

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        h_lst, s_lst, a_lst, r_lst, ns_lst, dn_lst = zip(*batch)
        '''
        B = batch_size, T = max_step, N = n_agents, H_dim = hidden_dim, obs_dim = observation_dim
        '''
        h_tensor = torch.stack(h_lst).to(self.device)     # (B, T+1, N, H_dim) 
        s_tensor = torch.stack(s_lst).to(self.device)     # (B, T, N, obs_dim)
        a_tensor = torch.stack(a_lst).long().unsqueeze(-1).to(self.device)      # action.shape = (B, T, N) -> (B, T, N, 1)
        r_tensor = torch.stack(r_lst).unsqueeze(-1).to(self.device)             # reward.shape = (B, T) -> (B, T, 1), q_tot
        ns_tensor = torch.stack(ns_lst).to(self.device)   # (B, T, N, obs_dim)
        d_tensor = torch.stack(dn_lst).unsqueeze(-1).to(self.device)            # dones.shape = (B, T, N) -> (B, T, N, 1)

        return h_tensor, s_tensor, a_tensor, r_tensor, ns_tensor, d_tensor

    def __len__(self):
        return len(self.buffer)