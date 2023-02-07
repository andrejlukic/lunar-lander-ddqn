import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import collections # For dequeue for the memory buffer
import random
import dill as pickle # For storing the buffer state

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class MemoryBuffer(object):
    def __init__(self, max_size):
        self.memory_size = max_size
        self.trans_counter=0 # num of transitions in the memory
                             # this count is required to delay learning
                             # until the buffer is sensibly full
        self.index=0         # current pointer in the buffer
        self.buffer = collections.deque(maxlen=self.memory_size)
        self.transition = collections.namedtuple("Transition", field_names=["state", "action", "reward", "new_state", "terminal"])

    
    def save(self, state, action, reward, new_state, terminal):
        t = self.transition(state, action, reward, new_state, terminal)
        self.buffer.append(t)
        self.trans_counter = (self.trans_counter + 1) % self.memory_size

    def random_sample(self, batch_size):
        assert len(self.buffer) >= batch_size # should begin sampling only when sufficiently full
        transitions = random.sample(self.buffer, k=batch_size) # number of transitions to sample
        states = torch.from_numpy(np.vstack([e.state for e in transitions if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in transitions if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in transitions if e is not None])).float().to(device)
        new_states = torch.from_numpy(np.vstack([e.new_state for e in transitions if e is not None])).float().to(device)
        terminals = torch.from_numpy(np.vstack([e.terminal for e in transitions if e is not None]).astype(np.uint8)).float().to(device)
  
        return states, actions, rewards, new_states, terminals

class QNN(nn.Module):
    def __init__(self, state_size, action_size, seed):
        super(QNN, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_size)
        
    def forward(self, state):
        x = self.fc1(state)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        return self.fc3(x)

class Agent(object):
    def __init__(self, gamma=0.99, epsilon=1.0, batch_size=128, lr=0.001,
                 epsilon_dec=0.996,  epsilon_end=0.01,
                 mem_size=1000000):
        self.gamma = gamma # alpha = learn rate, gamma = discount
        self.epsilon = epsilon
        self.epsilon_dec = epsilon_dec # decrement of epsilon for larger spaces
        self.epsilon_min = epsilon_end
        self.batch_size = batch_size
        self.memory = MemoryBuffer(mem_size)

    def save(self, state, action, reward, new_state, done):
        # self.memory.trans_counter += 1
        self.memory.save(state, action, reward, new_state, done)

    def choose_action(self, state):
        # state = state[np.newaxis, :]
        rand = np.random.random()
        state = torch.from_numpy(state).float().unsqueeze(0)
        self.q_func.eval()
        with torch.no_grad():
            action_values = self.q_func(state)
        self.q_func.train()
        # print(state)
        if rand > self.epsilon: 
            return np.argmax(action_values.cpu().data.numpy())
        else:
            # exploring: return a random action
            return np.random.choice([i for i in range(4)])     

    def reduce_epsilon(self):
        self.epsilon = self.epsilon*self.epsilon_dec if self.epsilon > \
                       self.epsilon_min else self.epsilon_min  
        
    def learn(self):
        raise Exception("Not implemented")
        
    def save_model(self, path):
        torch.save(self.q_func.state_dict(), path)

    def load_saved_model(self, path):
        self.q_func = QNN(8, 4, 42).to(device)
        self.q_func.load_state_dict(torch.load(path))
        self.q_func.eval()
        
    
class DoubleQAgent(Agent):
    def __init__(self, gamma=0.99, epsilon=1.0, batch_size=128, lr=0.001,
                 epsilon_dec=0.996,  epsilon_end=0.01,
                 mem_size=1000000, replace_q_target = 100):
        
        super().__init__(lr=lr, gamma=gamma, epsilon=epsilon, batch_size=batch_size,
             epsilon_dec=epsilon_dec,  epsilon_end=epsilon_end,
             mem_size=mem_size)

        self.replace_q_target = replace_q_target
        self.q_func = QNN(8, 4, 42).to(device)
        self.q_func_target = QNN(8, 4, 42).to(device)
        self.optimizer = optim.Adam(self.q_func.parameters(), lr=lr)
        
        
    def learn(self):
        if self.memory.trans_counter < self.batch_size: # wait before you start learning
            return
            
        # 1. Choose a sample from past transitions:
        states, actions, rewards, new_states, terminals = self.memory.random_sample(self.batch_size)
        
        ## Compute and minimize the loss
        ### Extract next maximum estimated value from target network
        q_targets_next = self.q_func_target(new_states).detach().max(1)[0].unsqueeze(1)
        ### Calculate target value from bellman equation
        q_targets = rewards + self.gamma * q_targets_next * (1 - terminals)
        ### Calculate expected value from local network
        q_expected = self.q_func(states).gather(1, actions)
        
        ### Loss calculation (we used Mean squared error)
        loss = F.mse_loss(q_expected, q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        # self.soft_update(self.QNN_local, self.QNN_target, TAU) 
        
        # 5. Reduce the exploration rate
        self.reduce_epsilon()
        
        if self.memory.trans_counter % self.replace_q_target == 0: # wait before you start learning
            for target_param, local_param in zip(self.q_func_target.parameters(), self.q_func.parameters()):
                target_param.data.copy_(local_param.data)


    def save_model(self, path):
        super().save_model(path)
        torch.save(self.q_func.state_dict(), path+'.target')


    def load_saved_model(self, path):
        super().load_saved_model(path)
        self.q_func_target = QNN(8, 4, 42).to(device)
        self.q_func_target.load_state_dict(torch.load(path+'.target'))
        self.q_func_target.eval()
    