
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib.style.core import available

class GRUCell(nn.Module):
    r"""Plain vanilla policy gradient network."""

    def __init__(self, gru_input_size, gru_hidden_size, additional_input_size, fc_output_size):
        super(GRUCell, self).__init__()
        # Define GRU layer
        self.gru = nn.GRUCell(input_size=gru_input_size, hidden_size=gru_hidden_size)

        # Define the input dimensions of the second fully connected layer, including GRU output and additional input dimensions
        self.fc_input_size = gru_hidden_size + additional_input_size

        # Define fully connected layer
        self.fc = nn.Linear(self.fc_input_size, fc_output_size)

    def forward(self, gru_input, additional_input):
        # GRU forward propagation, obtaining output and the last hidden state
        gru_output = self.gru(gru_input)

        # Splicing GRU output and additional inputs
        combined_input = torch.cat((gru_output, additional_input), dim=-1)  # (batch_size, fc_input_size)

        # Fully connected layer forward propagation
        fc_output = self.fc(combined_input)

        return fc_output

class GRDQN:

    def __init__(self, num_devices, device_state_size, compute_state_size, hidden_dim, action_dim, learning_rate, gamma,
                 epsilon_start, epsilon_min, epsilon_decay, target_update, device):
        self.num_devices = num_devices
        self.device_state_size = device_state_size
        self.compute_state_size = compute_state_size
        self.action_dim = action_dim
        self.state_dim = num_devices*device_state_size
        self.q_net = GRUCell(self.state_dim, hidden_dim,compute_state_size,
                          self.action_dim).to(device)  # Q network
        # target network
        self.target_q_net = GRUCell(self.state_dim, hidden_dim,compute_state_size,
                                 self.action_dim).to(device)
        # Using Adam optimizer
        self.optimizer = torch.optim.Adam(self.q_net.parameters(),
                                          lr=learning_rate)
        self.gamma = gamma  # discount factor

        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        self.target_update = target_update  # Target network update frequency
        self.count = 0  # Counter, record the number of updates
        self.device = device

    def check_action(self, device_states, action):
        """
        Check if the operation is effective.
        This method first adjusts the device state to a shape that is easy to handle, and then checks whether the specified operation is within the valid device range,
        And whether the device is available (i.e. whether the last flag of the device status is 0).
        Parameters:
        - device_states (Tensor):  The status information of the device is expected to be a Tensor.
        - action (int):  The specified operation should correspond to a certain device.
        return:
        -Boolean: If the operation is valid (i.e. within the range and the device is available), return True; Otherwise, return False.
        """

        # Adjust the shape of the device status for subsequent processing
        device_states = device_states.view(self.num_devices, -1)

        # Check if the operation is within the valid range
        if action < 0 or action >= self.num_devices:
            return False

        # Check if the device for the specified operation is available
        if device_states[action, -1].item():
            return False

        return True

    def take_action(self, device_states, compute_state):
        """
        Use epsilon greedy strategy to select actions.
        Parameters:
        -DeviceStatus Tensor: Device Status Tensor.
        -Compute_state: Compute the state tensor.
        return:
        -The selected action.
        """
        # Random number less than epsilon, explore
        if np.random.random() < self.epsilon:
            # Reshaping device states for easier access to the status of each device
            device_states_re = device_states.view(self.num_devices,-1)
            # Filter out unoccupied devices
            available_device = [ i for i in range(self.num_devices) if not device_states_re[i,-1].item()]
            # Randomly select an available device as the action
            action =  available_device[np.random.randint(len(available_device))]

        else:
            # Convert the device state and computation state into a one-dimensional tensor and move it to the specified device
            device_states = device_states.view(-1).to(self.device)
            compute_state = compute_state.view(-1).to(self.device)
            # Using Q network to calculate action values
            action_out = self.q_net(device_states,compute_state)
            # Select the action with the highest Q value
            action = action_out.argmax().item()
            # Initialize counter
            i = 1
            # If the selected action is not feasible, choose the action with the second highest Q value until a feasible action is found
            while not self.check_action(device_states, action) and i <= self.action_dim:
                _,action_tensor = torch.topk(action_out,i)
                action = action_tensor[-1].item()
                i+=1

        return action

    def update(self, transition_dict):
        # Retrieve data from transition_dict
        device_states = torch.tensor(transition_dict['device_states'],
                                     dtype=torch.float).to(self.device)
        compute_states = torch.tensor(transition_dict['compute_states'],
                              dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'],
                               dtype=torch.float).view(-1, 1).to(self.device)
        next_device_states= torch.tensor(transition_dict['next_device_states'],
                              dtype=torch.float).to(self.device)
        next_compute_states = torch.tensor(transition_dict['next_compute_states'],
                              dtype=torch.float).to(self.device)

        dones = torch.tensor(transition_dict['dones'],
                             dtype=torch.float).view(-1, 1).to(self.device)

        q_values = self.q_net(device_states,compute_states).gather(1, actions)
        # The maximum Q value for the next state
        max_next_q_values = self.target_q_net(next_device_states,next_compute_states).max(1)[0].view(
            -1, 1)
        q_targets = rewards + self.gamma * max_next_q_values * (1 - dones
                                                                )
        dqn_loss = torch.mean(F.mse_loss(q_values, q_targets))  # Mean square error loss function
        self.optimizer.zero_grad()
        dqn_loss.backward()  # Reverse propagation updates parameters
        self.optimizer.step()

        # Attenuation ε value
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        if self.count % self.target_update == 0:
            self.target_q_net.load_state_dict(
                self.q_net.state_dict())  # Update target network
        self.count += 1