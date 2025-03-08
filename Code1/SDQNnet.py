import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque
# from tensorflow.keras.callbacks import TensorBoard

# 定义Q网络
class SDQNNet(nn.Module):
    """
    Initialize the SDQN network architecture.
    This network is a fully connected neural network consisting of three linear layers. It is mainly used to select the best action based on the current state.
    Parameters:
    - state_size (int):  Dimension of input state. This determines the size of the network input layer.
    - action_size (int):  The number of possible actions. This determines the size of the network output layer.
    """
    def __init__(self, state_size, action_size):
        # Call the initialization method of the parent class to ensure the correct initialization of the module
        super(SDQNNet, self).__init__()
        # Define network structure
        self.fc1 = nn.Linear(state_size, 160)
        self.fc2 = nn.Linear(160, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, action_size)  # Output layer
        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        # Initialize
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        Parameters:
        x (Tensor):  Input data tensor
        return:
        Tensor:  Output data tensor after forward propagation of the model
        """
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        # Finally, a linear transformation is performed through the fully connected layer fc4 to obtain the final output
        return self.fc4(x)

# Define the SDQN model
class SDQN:
    """
       Parameters:
        state_size (int):  The size of the state space.
        action_size (int):  The size of the action space.
        gamma (float):  Discount factor, used to calculate the present value of future rewards.
        lr (float):  The learning rate determines the speed of network parameter updates.
        Epsilon_start (float): The starting value of epsilon in the greedy strategy of epsilon.
        Epsilon-min (float): The minimum value of epsilon used for stable learning.
        Epsilon decay (float): The decay rate of epsilon, which gradually decreases over time and increases the conversion from exploration to utilization.
        memory_size (int):  Experience replay memory size.
        batch_size (int):  The sample size sampled from memory during each experience replay.
       """
    def __init__(self, state_size, action_size, gamma, lr, epsilon_start, epsilon_min, epsilon_decay, memory_size, batch_size):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.lr = lr
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.memory = deque(maxlen=memory_size)
        self.batch_size = batch_size

        # Initialize Q network and target network
        self.q_network = SDQNNet(state_size, action_size)
        self.q_target_network = SDQNNet(state_size, action_size)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.lr)

        # Synchronize the parameters of the Q target network with the parameters of the Q network
        self.q_target_network.load_state_dict(self.q_network.state_dict())
        self.q_target_network.eval()  # The target network does not participate in training

    def select_action(self, state):
        """
        Select an action based on the current state.
        Parameters:
        state (list/np.array):  Current environmental status.
        return:
        action (tuple):  The selected action is in the format of (stp, con).
        """
        # The greedy strategy of epsilon determines whether to explore or utilize
        if random.random() < self.epsilon:
            # Random exploration action
            action = self.random_valid_action()
        else:
            # Convert the state to a tensor and obtain the Q value
            state = torch.FloatTensor(state)
            q_values = self.q_network(state)
            # Select action based on maximum Q value
            action_idx = torch.argmax(q_values).item()
            # Convert action index to (stp, con) format
            action = self.index_to_action(action_idx)
        return action

    def random_valid_action(self):
        """
        Randomly generate a valid action to ensure that (stp, con) is not both 0 or 1 at the same time.
        """
        valid_actions = [(0, 1), (1, 0)]  # Only allow these two combinations
        return random.choice(valid_actions)

    def index_to_action(self, idx):
        """
        Map the action index to (stp, con) format, 0->(0,1), 1->(1,0).
        """
        action_map = [(0, 1), (1, 0)]
        return action_map[idx]

    def action_to_index(self, action):
        """
        Map actions (stp, con) to indexes.
        """
        action_map = {(0, 1): 0, (1, 0): 1}
        return action_map[action]

    def store_transition(self, state, action, reward, next_state, done):
        """
        Store a transition into memory.
        """
        self.memory.append((state, action, reward, next_state, done))

    def train(self):
        """
        Train the Q network using data from experience replay.
        """
        if len(self.memory) < self.batch_size:
            return  # If there is insufficient data in the experience pool, return directly

        # Randomly select a batch of data from the experience pool
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # Convert states, actions, rewards, next states, and completion flags into appropriate tensor formats
        states_array = np.array(states)
        states = torch.FloatTensor(states_array)
        actions = torch.LongTensor([self.action_to_index(a) for a in actions]).unsqueeze(1)  # Map (stp, con) to an index
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        next_states_array = np.array(next_states)
        next_states = torch.FloatTensor(next_states_array)
        dones = torch.FloatTensor(dones).unsqueeze(1)

        # Calculate the Q value in the current state
        current_q = self.q_network(states).gather(1, actions)

        # Calculate the target Q value and use the target network
        next_q = self.q_target_network(next_states).max(1)[0].unsqueeze(1)
        target_q = rewards + (self.gamma * next_q * (1 - dones))
        loss = nn.MSELoss()(current_q, target_q)

        # Backpropagation, updating network weights
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Attenuation ε value
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def update_target_network(self):
        """
        Synchronize target network parameters.
        """
        self.q_target_network.load_state_dict(self.q_network.state_dict())
