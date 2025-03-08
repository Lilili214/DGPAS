import math
import torch
import collections
import random
import numpy as np

from utils import get_runtime_data,get_one_step_execution_time,get_execution_time,map_device_id_to_name

# 定义设备类
class Device:
    """
    设备类，用于表示一个具有特定规格的设备。
    参数:
    - memory (float): 设备的内存大小，记为M_p。
    - bandwidth (float): 设备的带宽大小，记为B_p。
    - compute_capacity (float): 设备的计算能力，记为F_p。
    属性:
    - assigned (bool): 表示设备是否已被分配用于某个任务或处于未分配状态。
    """
    def __init__(self, memory, bandwidth, compute_capacity):
        self.memory = memory  # 内存 M_p
        self.bandwidth = bandwidth  # 带宽 B_p
        self.compute_capacity = compute_capacity  # 计算能力 F_p
        self.assigned = False  # 是否已分配阶段


    def get_state(self):
        return torch.tensor([
            self.memory,
            self.bandwidth,
            self.compute_capacity,
            float(self.assigned)
        ], dtype=torch.float32)

# 首先定义经验回放池的类，主要包括加入数据、采样数据两大函数
class ReplayBuffer:
    ''' 经验回放池 '''
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)  # 队列,先进先出

    def add(self, device_states, compute_state, action, reward, next_device_states, next_compute_state, done):  # 将数据加入buffer
            self.buffer.append((device_states, compute_state, action, reward, next_device_states, next_compute_state, done))

    def sample(self, batch_size):  # 从buffer中采样数据,数量为batch_size
        transitions = random.sample(self.buffer, batch_size)
        device_states, compute_states, action, reward, next_device_states, next_compute_states, done = zip(*transitions)
        return np.array(device_states), np.array(compute_states), action, reward, np.array(next_device_states), np.array(next_compute_states), done

    def size(self):  # 目前buffer中数据的数量
        return len(self.buffer)

# 定义阶段的计算需求
class ComputeRequirement:
    def __init__(self, compute_amount, memory_requirement, comm_requirement):
        """
        初始化计算需求类
        参数:
        compute_amount (float): 计算量 W_r
        memory_requirement (float): 内存需求 M_r
        """
        self.compute_amount = compute_amount  # 计算量 W_r
        self.memory_requirement = memory_requirement  # 内存需求 M_r
        self.comm_requirement = comm_requirement #通信需求

# 定义环境类
class Environment:
    """
    环境类，用于模拟设备资源分配和计算任务调度问题。
    """
    def __init__(self, devices, compute_requirements, max_steps, batch_size, minibatch_size, runtime_data, stages):
        """
        初始化环境。
        参数:
        - devices (list): 设备列表
        - compute_requirements (list): 计算需求列表
        - max_steps (int): 最大步骤数
        """
        self.devices = devices
        self.compute_requirements = compute_requirements
        self.max_steps = max_steps
        self.batch_size = batch_size
        self.minibatch_size = minibatch_size
        self.max_stage_time = 0
        self.current_step = 0
        self.assignments = [None] * len(compute_requirements)  # 记录每个阶段分配的设备
        # 定义设备间的带宽 (双向)
        self.bandwidth = {}
        self.last_action = -1
        self.stage_comm_times = [0] * len(compute_requirements)
        self.stage_execution_times = [0] * len(compute_requirements)  # 添加这行，记录每个阶段的执行时间
        self.runtime_data = runtime_data
        self.stages = stages

    def calculate_execution_time(self, device, step):
        """
        计算在给定设备上执行特定计算需求的时间
        执行时间 = 计算量 / 计算能力
        """
        # compute_time = compute_requirement.compute_amount / device.compute_capacity
        compute_time = 0
        # print(device)
        # print(self.stages[step])
        for layer_index in self.stages[step]:
            compute_time += get_one_step_execution_time(self.runtime_data,map_device_id_to_name(device), layer_index)
        return compute_time

    def calculate_communication_time(self, device1_id, device2_id, compute_requirement):
        """
        计算在给定设备间执行特定计算需求的通信时间
        通信时间 = 内存需求 / 带宽
        """
        comm_time = 0
        if self.current_step == 0:
            return comm_time
        else:
            bw = self.bandwidth.get((device1_id, device2_id), 1e-6)
            comm_time = compute_requirement.comm_requirement / bw
            return comm_time



    def reset(self):
        """
        重置环境。
        返回:
        - state (Tensor): 初始状态
        """
        self.current_step = 0
        self.max_stage_time = 0
        self.assignments = [None] * len(self.compute_requirements)
        self.stage_execution_times = [0] * len(self.compute_requirements)  # 重置执行时间
        for device in self.devices:
            device.assigned = False
        device_states = torch.stack([device.get_state() for device in self.devices]).view(-1)

        self.last_action = -1
        self.stage_comm_times = [0] * len(self.compute_requirements)
        for d1 in range(len(self.devices)):
            for d2 in range(len(self.devices)):
                if d1 != d2:
                    # 计算带宽为 memory_bandwidth 的最小值，用于表示设备间的通信带宽
                    self.bandwidth[(d1, d2)] = min(self.devices[d1].bandwidth, self.devices[d2].bandwidth)

        compute_state = torch.tensor([
            self.compute_requirements[0].compute_amount,
            self.compute_requirements[0].memory_requirement
        ], dtype=torch.float32).view(-1)
        return  device_states, compute_state


    def step(self, action):
        """
        执行一个动作。
        参数:
        - action (int): 动作，表示分配给当前阶段的设备索引
        返回:
        - next_state (Tensor): 下一个状态
        - reward (float): 奖励
        - done (bool): 是否完成标志
        """

        # 分配设备
        self.devices[action].assigned = True
        self.assignments[self.current_step] = action

        #计算并记录通信时间
        comm_time = self.calculate_communication_time(
            action,
            self.last_action,
            self.compute_requirements[self.current_step]
        )
        self.stage_comm_times[self.current_step] = comm_time


        # 计算并记录执行时间
        execution_time = self.calculate_execution_time(
            action,
            self.current_step
        )
        self.stage_execution_times[self.current_step] = execution_time

        self.max_stage_time = max(self.max_stage_time, execution_time)

        # 计算奖励
        reward = self.calculate_reward(action)

        # 更新状态
        self.current_step += 1
        next_device_states = torch.stack([device.get_state() for device in self.devices]).view(-1)
        if self.current_step < len(self.compute_requirements):
            next_compute_state = torch.tensor([
            self.compute_requirements[self.current_step].compute_amount,
            self.compute_requirements[self.current_step].memory_requirement
        ], dtype=torch.float32).view(-1)
        else :
            next_compute_state = torch.tensor([
            self.compute_requirements[0].compute_amount,
            self.compute_requirements[0].memory_requirement
        ], dtype=torch.float32).view(-1)

        self.last_action = action

        # 检查是否完成
        done = self.current_step >= self.max_steps or self.current_step >= len(self.compute_requirements)

        return next_device_states, next_compute_state, reward, done

    def calculate_reward(self, action):
        """
        优化后的奖励函数，最小化总执行时间
        参数:
        - action (int): 动作，表示分配给当前阶段的设备索引
        返回:
        - reward (float): 奖励
        """
        # 当前阶段的计算需求
        compute_requirement = self.compute_requirements[self.current_step]
        device = self.devices[action]  # 分配的设备

        # 动态权重系数
        alpha = 0.6  # 延迟的权重
        beta = 0.2  # 负载均衡的权重
        gamma = 20  # 满足需求的正向奖励权重

        # 计算执行时间和通信时间
        # execution_time = compute_requirement.compute_amount / (device.compute_capacity + 1e-6)  # 避免除零
        execution_time = self.stage_execution_times[self.current_step]
        communication_time = 0
        if self.current_step > 0:
            communication_time = compute_requirement.comm_requirement / (
                self.bandwidth.get((action, self.last_action), 1e-6)  # 避免除零
            )

        # 标准化延迟 (执行时间 + 通信时间)
        total_delay = execution_time + communication_time
        normalized_delay = total_delay / self.max_steps
        delay_penalty = alpha * math.sqrt(normalized_delay)

        # 计算资源负载差异（考虑内存和计算能力）
        memory_diff = max(0, compute_requirement.memory_requirement - device.memory) / device.memory
        compute_diff = max(0, compute_requirement.compute_amount - device.compute_capacity) / device.compute_capacity
        # load_imbalance = math.sqrt(memory_diff ** 2 + compute_diff ** 2)
        load_imbalance = math.sqrt(abs(memory_diff + compute_diff))

        load_penalty = beta * load_imbalance

        # 正向激励：完全满足需求
        if (device.memory >= compute_requirement.memory_requirement and
                device.compute_capacity >= compute_requirement.compute_amount):
            positive_reward = gamma  # 满足所有需求的奖励
        else:
            positive_reward = 0

        # 综合奖励
        reward = #Please refer to the paper for the reward design

        return reward

    def get_total_execution_time(self):
        """
        计算所有阶段的总执行时间。
        """
        # total_time = 0
        # for i in range(len(self.compute_requirements)):
        #     if self.stage_execution_times[i] < self.stage_comm_times[i] :
        #         total_time += self.stage_comm_times[i]
        #     else:
        #         total_time = total_time

        total_stage_time = sum(self.stage_execution_times[i] + self.stage_comm_times[i]
                         for i in range(len(self.compute_requirements)))

        for i in range(self.minibatch_size - 1):
            total_stage_time += self.max_stage_time


        return total_stage_time


def moving_average(a, window_size):
    cumulative_sum = np.cumsum(np.insert(a, 0, 0))
    middle = (cumulative_sum[window_size:] - cumulative_sum[:-window_size]) / window_size
    r = np.arange(1, window_size-1, 2)
    begin = np.cumsum(a[:window_size-1])[::2] / r
    end = (np.cumsum(a[:-window_size:-1])[::2] / r)[::-1]
    return np.concatenate((begin, middle, end))