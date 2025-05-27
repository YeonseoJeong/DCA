import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from MARL_DCA.env.qmix2 import QMIX, ReplayBufferRNN
import torch.nn.functional as F
import time

if 'ipykernel' in sys.modules:
    from IPython import display


class AccessPoint: 
    """Access Point for DCA of CSMA/CA, IPPO, QLBT"""
    SIF_DURATION = 2
    STATE_ACK = "ACK"
    STATE_BUSY = "BUSY"
    STATE_IDLE = "IDLE"
    STATE_COLLISION = "COLLISION"

    def __init__(self, ):
        self.channel_busy = False 
        self.current_transmitter = None   # ID of the node of current transmission 
        self.remaining_slots = 0          # Number of remaining slots for current transmission
        self.sif = self.SIF_DURATION
        self.delay = 0
        self.sif_mode = False

    def receive(self, nodes, packet_length):    
        ''' 
        Handle transmission attempt and return current channel state.
        Args:
            nodes (list): Nodes attempting to transmit
            packet_length (int): Packet length
        Returns:
            str: Channel state 
        '''
        if self.channel_busy and self.remaining_slots == 0 and self.sif == 0:
            # print(f"[DEBUG] receive() | remaining_slots={self.remaining_slots}, sif={self.sif}, busy={self.channel_busy}")
            self.sif_mode = False
            self.channel_busy = False
            return self.STATE_ACK
        
        # busy channel
        if self.channel_busy:    
            # if len(nodes) > 0: 
            #     # self.reset()
            #     # return self.STATE_COLLISION
            #     return self.STATE_BUSY
            
            if self.remaining_slots > 0:
                return self.STATE_BUSY
            
            if self.remaining_slots == 0 and self.sif > 0:
                self.sif_mode = True
                return self.STATE_BUSY
            
        
        # idle channel
        if len(nodes) > 1:
            self.delay += 1
            return self.STATE_COLLISION
        elif len(nodes) == 1: 
            self.start_transmission(nodes[0], packet_length) 
            return self.STATE_BUSY
        else: # len(nodes) == 0
            self.delay += 1
            return self.STATE_IDLE
        
    def start_transmission(self, node_id, packet_length):
        """Start transmission for a node"""
        self.channel_busy = True 
        self.current_transmitter = node_id
        self.remaining_slots = packet_length
        self.sif = self.SIF_DURATION
            
    def update(self): 
        """Decrement remaining slots or SIF"""
        # if self.channel_busy:     
        # # print(f"[DEBUG] update() | remaining_slots={self.remaining_slots}, sif={self.sif}")
        #     if self.remaining_slots > 0:
        #         self.remaining_slots -= 1
            
        #     if self.remaining_slots == 0 and self.sif > 0: 
        #         self.sif -= 1 
        if self.channel_busy and self.remaining_slots > 0:
            self.remaining_slots -= 1 
        if self.remaining_slots == 0 and self.sif > 0: 
            self.sif -= 1 

    def reset(self):
        """Reset channel """
        self.channel_busy = False 
        self.current_transmitter = None 
        self.remaining_slots = 0 
        self.sif = self.SIF_DURATION


class QLBT_DCAEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array", "ansi"], "render_fps": 10}  
    RENDER_SLOTS = 200
    NOTEBOOK = 'ipykernel' in sys.modules 
    CHANNEL_STATE_MAP = {
        "IDLE": 0,
        "BUSY": 1,
        "COLLISION": 2,
        "ACK": 3
    }

    def __init__(self, num_agents=5, max_cycles = 1000, packet_length =3, render_mode=None):
        super().__init__()
        self.num_agents = num_agents
        self.max_cycles = max_cycles
        self.packet_length = packet_length
        self.render_mode = render_mode.lower() if render_mode else None

        self.access_point = AccessPoint()
        self.t = 0

        self.agents = [f"agent_{i}" for i in range(num_agents)] # ["agent_0", "agent_1", ...]
        self.action_space = {a: gym.spaces.Discrete(2) for a in self.agents} # 0: wait, 1: transmit
        self.observation_space = {
            a: gym.spaces.Dict({
                "channel_state": gym.spaces.Discrete(4), # 0: IDLE, 1: BUSY, 2: COLLISION, 3: ACK
                "collision": gym.spaces.Discrete(2), # 0: no collision, 1: collision
                "own_d2lt": gym.spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
            })
            for a in self.agents
        }

        self.hidden_state = {
            "channel": "IDLE",
            "ready_nodes": [],
            "D2LT": [0] * self.num_agents, # 각 agent의 D2LT
            "others": [0] * self.num_agents, # 다른 사람의 데이터 전송 여부, 논문 상 O_t
            "successful": False
        }

        self._initialize_render_data()

    def reset(self, seed = None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)

        self.t = 0
        self.access_point.reset()
        self.hidden_state = {
            "channel": "IDLE",
            "ready_nodes": [],
            "successful": False,
            "D2LT": [0] * self.num_agents,
            "others": [0] * self.num_agents
        }
        self._initialize_render_data()
        obs = self._get_obs() # 각 agent의 관측값
        infos = {a:{"hidden_state": self.hidden_state.copy()} for a in self.agents}

        return obs, infos

    def step(self, actions): 
        # print(f"[DEBUG] step 시작 | t={self.t}, channel={self.hidden_state.get('channel', None)}")
        # if self.hidden_state["channel"] == "ACK":
        #     actions = {a: 0 for a in self.agents} # 모든 agent는 대기 상태로 전환
        #     ready_nodes = []
        # else: 
        action_array = np.array([actions[a] for a in self.agents])
        ready_nodes = np.where(action_array == 1)[0].tolist()
    
        channel = self.access_point.receive(ready_nodes, self.packet_length)
        self.hidden_state["channel"] = channel
        self.hidden_state["ready_nodes"] = ready_nodes

        for i in range(self.num_agents):
            self.hidden_state["D2LT"][i] += 1
            self.hidden_state["others"][i] = int(actions[self.agents[i]] == 1)

        # Update the ongoing packet transmission 
        if channel in ["BUSY", "ACK"]: 
            self.hidden_state["successful"] = True 
        elif channel == "COLLISION": 
            self.hidden_state["successful"] = False 

        rewards = self._compute_reward() 
        self.access_point.update() 
        self.t += 1
        observations = self._get_obs()

        truncations = {a: self.t >= self.max_cycles for a in self.agents}
        terminations = {a: False for a in self.agents}

        infos = {a: {"hidden_state": self.hidden_state.copy()} for a in self.agents}

        self._update_render_data()

        return observations, rewards, terminations, truncations, infos

    # def get_state(self):
    #     channel_state = self.CHANNEL_STATE_MAP[self.access_point.channel_state]
    #     return np.concatenate((
    #         [channel_state],
    #         self.success_count/(self.t+1e-5),
    #         self.joint_action, 
    #         self.d2lt
    #     ))
    
    def _get_obs(self): 
        obs = {}
        state = self.CHANNEL_STATE_MAP[self.hidden_state["channel"]]
        collision = int(self.hidden_state["channel"] == "COLLISION")

        # Normalize D2LT by max_cycles
        d2lt = [d / self.max_cycles for d in self.hidden_state["D2LT"]]

        for i, a in enumerate(self.agents):
            obs[a] = {
                "channel_state": state,
                "collision": collision,
                "own_d2lt": np.array([d2lt[i]], dtype=np.float32)
            }

        return obs
    
    def _compute_reward(self):
        # print("[DEBUG] _compute_reward() 호출됨")
        rewards = {a: 0.0 for a in self.agents}
        ready_nodes = self.hidden_state["ready_nodes"]
        channel = self.hidden_state["channel"]
        d2lt = self.hidden_state["D2LT"]
        self.render_data["time_slot"] += 1

        if channel == "ACK":
            winner = self.access_point.current_transmitter # idx
            self.render_data['success'] += 1
            self.render_data['agent_success'][winner] += 1
            self.render_data['success_packet'] += (self.packet_length + self.access_point.sif)
            self.render_data['agent_success_packet'][winner] += (self.packet_length + self.access_point.sif)

            max_d2lt_agent_idx = np.argmax(d2lt)

            for i, a in enumerate(self.agents):
                # rewards[a] = (self.hidden_state["D2LT"][i] * 10 / self.max_cycles)
                if i == winner:
                    if i == max_d2lt_agent_idx:
                        rewards[a] = 1.0
                    else:
                        rewards[a] = 1.0 * (d2lt[i] / sum(d2lt) + 1e-6)
                
            self.hidden_state["D2LT"][winner] = 0
            self.hidden_state["others"][winner] = 0
        
        elif channel == "COLLISION":
            self.render_data['collision'] += 1
            for node in ready_nodes:
                self.render_data['agent_collision'][node] += 1
                rewards[self.agents[node]] = -1.0

        return rewards
    
    def _initialize_render_data(self):
        self.render_data = {
            "time_agent": np.zeros((self.num_agents, self.RENDER_SLOTS)), # [3, 200]
            "agent_success": np.zeros((self.num_agents, 1), dtype=int),   # [3, 1]
            "agent_collision": np.zeros((self.num_agents, 1), dtype=int), # [3, 1]
            "agent_success_packet": np.zeros((self.num_agents, 1), dtype=int), # [3, 1]
            "time_slot": 0,
            "success": 0,           # 성공 횟수
            "success_packet": 0,    # 전송 성공한 패킷 수
            "collision": 0,         # 충돌 횟수
            "cumsum_success": [],
            "cumsum_success_packet": [],
            "cumsum_collision": [],
            "throughput": [],
            "throughput_packet": [],
            "delay": []
        }
    
    def _update_render_data(self): 
        """Update the render data based on the current state"""
        self.state_data = np.zeros((self.num_agents, 1))
        ready_nodes = self.hidden_state["ready_nodes"] 
        channel = self.hidden_state["channel"] 

        if channel == "ACK":
            self.state_data[self.access_point.current_transmitter] = 3
        elif channel == "COLLISION":
            for node in ready_nodes:
                self.state_data[node] = 2
        elif channel == "BUSY":
            if self.access_point.remaining_slots > 0 and not self.access_point.sif_mode:
                self.state_data[self.access_point.current_transmitter] = 1
            elif self.access_point.remaining_slots > 0 and len(ready_nodes) > 0: 
                for node in ready_nodes: 
                    self.state_data[node] = 2 
        
        self.render_data['time_agent'] = np.concatenate((self.render_data['time_agent'][:,1:], self.state_data), axis=1)
        self.render_data['delay'].append(np.mean(self.hidden_state["D2LT"]))

        if self.render_mode in ["human", "rgb_array"]:
            # t_safe = max(self.t, 1)
            self.render_data["cumsum_success"].append(self.render_data["success"])
            self.render_data["cumsum_success_packet"].append(self.render_data["success_packet"])
            self.render_data["cumsum_collision"].append(self.render_data["collision"])
            self.render_data["throughput"].append(self.render_data["success"] / self.render_data["time_slot"])
            self.render_data["throughput_packet"].append(self.render_data["success_packet"] / self.render_data["time_slot"])        

        
    def render(self):
        if not self.render_mode:
            return
        
        data = self.render_data['time_agent']
        
        if self.render_mode == "ansi":
            self._render_ansi(data)
        elif self.render_mode in ["rgb_array", "human"]:
            self._render_human(data)
    
    def _render_ansi(self, time_node_data): 
        """Render the environment in ANSI mode""" 
        if self.NOTEBOOK: 
            display.clear_output(wait=True)  
        print("*" * 35 + f"{'RENDERING DCA(t = ' + str(self.t) + ')':^40}" + "*" * 35)
        for i, row in enumerate(time_node_data):
            print(f"Node {i+1} (D2LT={self.hidden_state['D2LT'][i]:>3d}): ", end=' ')
            for rc in row:
                CCOLOR = '\033[44m' if rc == 1 else '\033[101m' if rc == 2 else '\033[42m' if rc == 3 else '\33[7m'
                CEND = '\33[0m'
                print(f"{CCOLOR}{int(rc)}{CEND}", end=' ')
            print()
        print(f"Time Slot: {self.t}, "
                f"Success: {self.render_data['success']}, "
                f"Collision: {self.render_data['collision']}, "
                f"Throughput: {self.render_data['success_packet'] / self.t if self.t > 0 else 0:.3f}, " 
                # f"Mean Delay: {self.render_data['delay'][-1]}"
                f"Mean Delay: {np.mean(self.hidden_state['D2LT'])}"
                )
        print("*" * 110)
        print()
        time.sleep(0.1)

    def _render_human(self, time_node_data):
        plt.clf()
        cmap = colors.ListedColormap(['white', 'blue', 'red', 'lightgreen'])
        norm = colors.BoundaryNorm([0, 1, 2, 3, 4], cmap.N)

        plt.subplot(311)
        plt.title(f"QLBT DCA (t = {self.t}, throughput: {self.render_data['success_packet'] / self.render_data['time_slot']:.3f})")
        plt.imshow(time_node_data, cmap=cmap, norm=norm, aspect='auto')
        plt.ylabel("Agent ID")

        plt.subplot(323)
        plt.plot(self.render_data["cumsum_success_packet"], 'b-', label="Success")
        plt.plot(self.render_data["cumsum_collision"], 'r-', label="Collision")
        plt.legend()
        plt.grid(True)

        plt.subplot(324)
        plt.bar(range(self.num_agents), self.render_data["agent_success_packet"].squeeze())
        plt.ylabel("Success Count")
 

        plt.subplot(313)
        plt.plot(self.render_data["throughput_packet"], 'b-', label="Throughput Packet")
        plt.legend()
        plt.grid(True)

        if self.NOTEBOOK:
            display.clear_output(wait=True)
            display.display(plt.gcf())
        else:
            plt.pause(0.001)

    def save_render_data(self, save_dir="logs/render_data", episode=None):
        os.makedirs(save_dir, exist_ok=True)
        data = self.render_data

        # (1) 1D 리스트 시계열 키 → 한 줄씩 CSV에 저장
        csv_keys = [
            "cumsum_success", 
            "cumsum_success_packet", 
            "cumsum_collision", 
            "delay", 
            "throughput", 
            "throughput_packet"
        ]

        for key in csv_keys:
            values = data.get(key, [])
            filepath = os.path.join(save_dir, f"{key}.csv")

            if isinstance(values, list) and len(values) == self.max_cycles:
                with open(filepath, "a") as f:
                    line = ",".join(map(str, values))
                    f.write(line + "\n")
            else:
                print(f"[WARN] Skipped {key} for episode {episode}: length = {len(values)}")


# def obs_dict_to_vec(obs_dict):
#     return np.concatenate((
#         [obs_dict["channel_state"]],
#         [obs_dict["collision"]],
#         obs_dict["own_d2lt"]
#     )).astype(np.float32)

    
if __name__ == "__main__":

    num_agents = 3
    hidden_dim = 128    # QMIX agent GRU hidden 크기
    action_dim = 2      # 0: wait, 1: transmit
    max_episodes = 1000
    max_cycles = 500
    packet_length = 10
    render_mode = "rgb_array"

    env = QLBT_DCAEnv(
        num_agents = num_agents,
        max_cycles = max_cycles,
        packet_length = packet_length,
        render_mode = render_mode
    )


    trainer = QMIX(env=env, hidden_dims=hidden_dim, batch_size=64, buffer_capacity=10000, lr=0.0003, gamma=0.95,
        epochs=10, max_steps=max_cycles, log_dir="logs/qmix_dca_logs", plot_window=100,
        update_interval=100, device="cpu", tau=0.01, decay_ratio = 0.99
    )

    trainer.buffer = ReplayBufferRNN(capacity=10000, device="cpu")

    trainer.train(max_episode=max_episodes)

    if not env.NOTEBOOK:
        plt.show()


#     def evaluate_qmix(env, trainer, episodes=5, episode_index=None):
#         total_reward = 0
#         eval_log = []
#         for ep in range(episodes):
#             obs_dict, _ = env.reset()
#             obs = {agent: trainer.preprocess_observation(obs_dict[agent], agent) for agent in env.agents}
#             h_states = {agent: torch.zeros(hidden_dim) for agent in env.agents}
#             last_actions = {agent: torch.zeros(action_dim) for agent in env.agents}
#             episode_reward = 0

#             for _ in range(max_cycles):
#                 actions, h_next = {}, {}
#                 for agent in env.agents:
#                     action, h_new = trainer.select_action(
#                         agent=agent,
#                         obs=obs[agent],
#                         last_action=last_actions[agent],
#                         his_in=h_states[agent],
#                         epsilon=0.0
#                     )
#                     actions[agent] = action
#                     h_next[agent] = h_new.detach()

#                 obs_next_dict, rewards, terminations, truncations, _ = env.step(actions)
#                 env.render()
#                 episode_reward += sum(rewards.values())
#                 obs = {agent: trainer.preprocess_observation(obs_next_dict[agent], agent) for agent in env.agents}
#                 h_states = h_next
#                 last_actions = {
#                     agent: F.one_hot(torch.tensor(actions[agent]), num_classes=action_dim).float() for agent in env.agents
#                 }                
#                 if all(terminations.values()) or all(truncations.values()):
#                     break

#             print(f"[EVAL] Episode {ep+1}: Total reward = {episode_reward:.2f}")
#             eval_log.append(episode_reward)
#             total_reward += episode_reward

#         avg_reward = total_reward / episodes
#         print(f"[EVAL] Average reward over {episodes} episodes: {avg_reward:.2f}")



#         trainer.train()
#         print(f"Episode {episode + 1}/{max_episodes} finished. Total reward: {sum([r.item() for r in r_seq]):.3f}, Steps: {len(r_seq)}")
#         env.save_render_data(env, save_dir="logs/render_data", episode=episode)

#         if (episode + 1) % 100 == 0:
#             trainer.save(f"checkpoints/qmix_ep{episode+1}.pth")
#             evaluate_qmix(env, trainer, episodes=5, episode_index=episode+1)

#     plt.show()
