from keras.layers import Dense, Activation
from keras.models import Sequential, load_model
from keras.optimizers import Adam
import numpy as np
import collections # For dequeue for the memory buffer
import random
import dill as pickle # For storing the buffer state


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

        states = np.array([e.state for e in transitions if e is not None]).astype(float)
        actions = np.array([e.action for e in transitions if e is not None]).astype(int)
        rewards = np.array([e.reward for e in transitions if e is not None]).astype(int)
        new_states = np.array([e.new_state for e in transitions if e is not None]).astype(float)
        terminals = np.array([e.terminal for e in transitions if e is not None]).astype(int)
  
        return states, actions, rewards, new_states, terminals

    
class Agent(object):
    def __init__(self, lr, gamma, epsilon, batch_size,
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
        state = state[np.newaxis, :]
        rand = np.random.random()
        if rand > self.epsilon: 
            # greedy, returning best known action
            sa = self.q_func.predict(state, verbose=0)
            return np.argmax(sa)            
        else:
            # exploring: return a random action
            return np.random.choice([i for i in range(4)])
            

    def reduce_epsilon(self):
        self.epsilon = self.epsilon*self.epsilon_dec if self.epsilon > \
                       self.epsilon_min else self.epsilon_min
        
        
    def learn(self):
        raise Exception("Not implemented")
        

    def save_model(self, path):
        self.q_func.save(path)
        with open(path + '.memory.pickle', 'wb') as handle:
            pickle.dump(self.memory, handle, protocol=pickle.HIGHEST_PROTOCOL)



    def load_saved_model(self, path):
        self.q_func = load_model(path)
        with open(path + '.memory.pickle', 'rb') as handle:
            self.memory = pickle.load(handle)
        
    
class SingleQAgent(Agent):
    def __init__(self, lr, gamma, epsilon, batch_size,
                 epsilon_dec=0.996,  epsilon_end=0.01,
                 mem_size=1000000):
        super().__init__(lr=lr, gamma=gamma, epsilon=epsilon, batch_size=batch_size,
                 epsilon_dec=epsilon_dec,  epsilon_end=epsilon_end,
                 mem_size=mem_size)

        nn = Sequential([
                    Dense(64, input_shape=(8,)),
                    Activation('relu'),
                    Dense(64),
                    Activation('relu'),
                    Dense(64),
                    Activation('relu'),
                    Dense(4)])
        nn.compile(optimizer=Adam(lr=lr), loss='mse')
        self.q_func = nn
        
        
    def learn(self):
        if self.memory.trans_counter < self.batch_size: # wait before you start learning
            return
            
        # 1. Choose a sample from past transitions:
        states, actions, rewards, new_states, terminals = self.memory.random_sample(self.batch_size)
        
        # 2. Compute predicted q value for the sample states
        q = self.q_func.predict(states, verbose=0)
        
        # 3. Compute (using the same Q network) q value for the new states
        q_next = self.q_func.predict(new_states, verbose=0)
        
        # 4. Improve the Q network
        inx = np.arange(self.batch_size, dtype=np.int32)
        q[inx, actions] = rewards + self.gamma*np.max(q_next, axis=1)*(1-terminals)
        self.q_func.fit(states, q, verbose=0)
        
        # 5. Reduce the exploration rate
        self.reduce_epsilon()

        
class DoubleQAgent(Agent):
    def __init__(self, lr, gamma, epsilon, batch_size,
                 epsilon_dec=0.996,  epsilon_end=0.01,
                 mem_size=1000000, replace_q_target = 100):
        
        super().__init__(lr=lr, gamma=gamma, epsilon=epsilon, batch_size=batch_size,
             epsilon_dec=epsilon_dec,  epsilon_end=epsilon_end,
             mem_size=mem_size)

        self.replace_q_target = replace_q_target
        nn = Sequential([
                    Dense(64, input_shape=(8,)),
                    Activation('relu'),
                    Dense(64),
                    Activation('relu'),
                    Dense(4)])
        nn.compile(optimizer=Adam(lr=lr), loss='mse')
        self.q_func = nn
        
        nnt = Sequential([
                    Dense(64, input_shape=(8,)),
                    Activation('relu'),
                    Dense(64),
                    Activation('relu'),
                    Dense(4)])
        nnt.compile(optimizer=Adam(lr=lr), loss='mse')
        self.q_func_target = nnt
        
        
    def learn(self):
        if self.memory.trans_counter < self.batch_size: # wait before you start learning
            return
            
        # 1. Choose a sample from past transitions:
        states, actions, rewards, new_states, terminals = self.memory.random_sample(self.batch_size)
        
        # 2. Compute predicted q value for the sample states
        q = self.q_func.predict(states, verbose=0)
        
        # 3. Compute (using the same Q network) q value for the new states
        q_next = self.q_func.predict(new_states, verbose=0)
        
        # 4. Update the Q target using the second Q network
        q_target = self.q_func_target.predict(new_states, verbose=0)
        
        # 4. Improve the Q network
        inx = np.arange(self.batch_size, dtype=np.int32)
        q[inx, actions] = rewards + self.gamma*q_target[inx, np.argmax(q_next, axis=1).astype(int)]*(1-terminals)
        self.q_func.fit(states, q, verbose=0)
        
        # 5. Reduce the exploration rate
        self.reduce_epsilon()
        
        if self.memory.trans_counter % self.replace_q_target == 0: # wait before you start learning
            self.q_func_target.set_weights(self.q_func.get_weights())


    def save_model(self, path):
        super().save_model(path)
        self.q_func_target.save(path+'.target')


    def load_saved_model(self, path):
        super().load_saved_model(path)
        self.q_func_target = load_model(path+'.target')
    