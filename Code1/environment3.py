import numpy as np
import pandas as pd
from utils import get_one_step_execution_time,get_runtime_data

# Environmental interaction section
class EnvSimulator:
    def __init__(self, total_time, total_layers, layer_resources, total_resources, devices, edge_devices, runtime):
        self.total_time = total_time
        self.total_layers = total_layers
        self.total_resources = total_resources
        self.layer_resources = layer_resources
        self.done = False
        self.devices = devices
        self.edge_devices = edge_devices
        self.phases = [list(range(self.total_layers))]
        self.current_phase_index = 0
        self.current_layer_index = 0
        self.phase_times = []
        self.compute_requirements = []
        self.current_state = self._initialize_state()
        self.runtime = runtime
        self.device_FLOPs = min([float(device["FLOPs"]) for _, device in self.devices.items()])
        self.bw = min([float(device["memory_bandwidth"]) for _, device in self.devices.items()])

    def _initialize_state(self):
        if self.done:
            return np.zeros(8)  # Status dimension

        current_phase = self.phases[self.current_phase_index]
        current_layer = current_phase[self.current_layer_index]
        resources = list(self.layer_resources[current_layer].values())
        total_resources = list(self.total_resources.values())
        state = np.array(resources + total_resources)
        return state

    def reset(self):
        self.phases = [list(range(self.total_layers))]
        self.current_phase_index = 0
        self.current_layer_index = 0
        self.done = False
        self.phase_times = []
        self.compute_requirements = []
        self.current_state = self._initialize_state()
        return self.current_state

    def compute_phase_time(self, phase_layers):
        """
        Calculate the total time of a stage.
        """
        compute_time = 0
        comm_time = 0

        for layer in phase_layers:
            compute_time += max(get_one_step_execution_time(self.runtime, 'jetson1', layer),
                                get_one_step_execution_time(self.runtime, 'jetson2', layer),
                                get_one_step_execution_time(self.runtime, 'raspberry', layer))

        comm_time = self.layer_resources[phase_layers[-1]]["MemR+W(B)"] / self.bw if phase_layers else 0

        return compute_time + comm_time

    def calculate_process_reward(self, stp, phase_layers, current_layer):
        """
        Rewards during the calculation process.
        """
        reward = 0
        if not phase_layers:  # Avoid empty lists
            return reward

        # Calculate the computational workload of the current stage
        current_phase_compute = sum([self.layer_resources[l]["Flops"] for l in phase_layers])
        # Calculate the current stage of communication volume
        current_phase_comm = self.layer_resources[phase_layers[-1]]["MemR+W(B)"]

        """
        Please refer to the paper for the reward design
        """
        # If segmentation (stp=1), check if the computational load before segmentation is too large
        if stp == 1:
            if current_phase_compute > self.device_FLOPs * 0.5:
                reward +=   # Encourage the segmentation of large computational stages
            else:
                reward -=

        # If not divided (con=1), check the communication volume
        else:
            if current_phase_comm > self.bw * 0.5:
                reward -=   # Do not encourage separating layers with high communication requirements
            else:
                reward +=

        return reward

    def calculate_final_reward(self):
        """
        Calculate the final reward.
        """
        if not self.phase_times:
            return 0

        T_avg = sum(self.phase_times) / len(self.phase_times)
        phase_variance = np.var(self.phase_times)
        balance_threshold = 1
        reward = 0

        if phase_variance <= balance_threshold:
            reward =  #Please refer to the paper for the reward design
        else:
            reward =  #Please refer to the paper for the reward design

        if all(t > T_avg * 0.9 for t in self.phase_times):
            reward +=  #Please refer to the paper for the reward design

        reward -=  #Please refer to the paper for the reward design
        return reward

    def step(self, action):
        stp, con = action
        if self.done:
            raise Exception("Environment is done.")

        current_phase = self.phases[self.current_phase_index]
        current_layer = current_phase[self.current_layer_index]

        # Reward during the calculation process
        process_reward = self.calculate_process_reward(stp, current_phase, current_layer)

        if stp == 1 and len(self.phases) < len(self.edge_devices):
            split_layer_index = self.current_layer_index
            phase_up_to_split = current_phase[:split_layer_index + 1]
            remaining_layers = current_phase[split_layer_index + 1:]

            self.phases[self.current_phase_index] = phase_up_to_split
            if remaining_layers:
                self.phases.insert(self.current_phase_index + 1, remaining_layers)

            phase_time = self.compute_phase_time(phase_up_to_split)
            self.phase_times.append(phase_time)

            self.current_phase_index += 1
            self.current_layer_index = 0

        if con == 1:
            if self.current_layer_index + 1 < len(current_phase):
                self.current_layer_index += 1
            else:
                self.current_phase_index += 1
                self.current_layer_index = 0

        if self.current_phase_index >= len(self.phases):
            self.done = True
            self.current_state = np.zeros(len(self.current_state))
            # Calculate the final reward
            final_reward = self.calculate_final_reward()
            reward = final_reward
        else:
            self.current_state = self._initialize_state()
            reward = process_reward

        return self.current_state, reward, self.done

    def return_phases(self):
        return self.phases

    def save_compute_requirement(self, phases):
        self.compute_requirements = []
        for phase in phases:
            compute_amount = sum([self.layer_resources[layer]["Flops"] for layer in phase])
            memory_requirement = sum([self.layer_resources[layer]["memory(MB)"] for layer in phase])
            comm_requirement = self.layer_resources[phase[-1]]["MemR+W(B)"] if phase else 0
            self.compute_requirements.append([compute_amount, memory_requirement, comm_requirement])

        cr_pd = pd.DataFrame(data=self.compute_requirements, index=None,
                             columns=['compute_amount', 'memory_requirement', 'comm_requirement'])
        cr_pd.to_csv('compute_requirements.csv')

    def get_total_training_time(self):
        return self.total_time