# -*- coding: utf-8 -*-
import math
import numpy as np
import torch
from collections import deque
import torch.nn as nn
from torch.autograd import Variable

import argparse
BATCH_SIZE=50;
def pdf(x,mu,sigma): 
    #mu 平均值，sigma:标准差
    return 1/(sigma*np.sqrt(2*np.pi))*np.exp(-(x-mu)**2/(2*sigma**2))
    #概率密度函数
    
    
#获取某一冰壶距离营垒圆心的距离
def get_dist(x, y):
    House_x = 2.375
    House_y = 4.88
    return math.sqrt((x-House_x)**2+(y-House_y)**2)

#根据冰壶比赛服务器发送来的场上冰壶位置坐标列表获取得分情况并生成信息状态数组
def get_infostate(position):
    House_R = 1.830
    Stone_R = 0.145

    init = np.empty([8], dtype=float)
    gote = np.empty([8], dtype=float)
    both = np.empty([16], dtype=float)
    #计算双方冰壶到营垒圆心的距离
    for i in range(8):
        init[i] = get_dist(position[4 * i], position[4 * i + 1])
        both[2*i] = init[i] 
        gote[i] = get_dist(position[4 * i + 2], position[4 * i + 3])
        both[2*i+1] = gote[i]
    #找到距离圆心较远一方距离圆心最近的壶
    if min(init) <= min(gote):
        win = 0                     #先手得分
        d_std = min(gote)
    else:
        win = 1                     #后手得分
        d_std = min(init)
    
    infostate = []  #状态数组
    init_score = 0  #先手得分
    #16个冰壶依次处理
    for i in range(16):
        x = position[2 * i]         #x坐标
        y = position[2 * i + 1]     #y坐标
        dist = both[i]              #到营垒圆心的距离
        sn = i % 2 + 1              #投掷顺序
        if (dist < d_std) and (dist < (House_R+Stone_R)) and ((i%2) == win):
            valid = 1               #是有效得分壶
            #如果是先手得分
            if win == 0:
                init_score = init_score + 1
            #如果是后手得分
            else:
                init_score = init_score - 1
        else:
            valid = 0               #不是有效得分壶
        #仅添加有效壶
        if x!=0 or y!=0:
            infostate.append([x, y, dist, sn, valid])
    #按dist升序排列
    infostate = sorted(infostate, key=lambda x:x[2])
    
    #无效壶补0
    for i in range(16-len(infostate)):
        infostate.append([0,0,0,0,0])

    #返回先手得分和转为一维的状态数组
    return init_score, np.array(infostate).flatten()

#创建Actor网络类继承自nn.Module
class Actor(nn.Module):
    def __init__(self):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(80, 128)       # 定义全连接层1
        self.fc2 = nn.Linear(128, 64)       # 定义全连接层2
        self.fc3 = nn.Linear(64, 10)        # 定义全连接层3
        self.out = nn.Linear(10, 3)         # 定义输出层
        self.out.weight.data.mul_(0.1)      # 初始化输出层权重

    def forward(self, x):
        x = self.fc1(x)                     # 输入张量经全连接层1传递
        x = torch.tanh(x)                   # 经tanh函数激活
        x = self.fc2(x)                     # 经全连接层2传递
        x = torch.relu(x)                   # 经tanh函数激活
        x = self.fc3(x)                     # 经全连接层3传递
        x = torch.tanh(x)                   # 经tanh函数激活

        mu = self.out(x)                    # 经输出层传递得到输出张量
        logstd = torch.zeros_like(mu)       # 生成shape和mu相同的全0张量
        std = torch.exp(logstd)             # 生成shape和mu相同的全1张量
        return mu, std, logstd

    def choose_action(self, state):
        x = torch.FloatTensor(state)  # 将输入状态数组转换为张量 shape: torch.Size([80])
        mu, std, _ = self.forward(x)  # 进行前向推理获取 mu 和 std，形状分别为 torch.Size([3]) 和 torch.Size([3])
        action = torch.normal(mu, std).data.numpy()  # 根据给定的均值和方差生成输出张量的近似数据

        action[0] = np.random.normal(loc=3, scale=0.25)  # 使用 numpy 生成在指定均值和方差下的随机数作为 action[0]
        action[0] = np.clip(action[0], 2.65, 3.4)  # 对 action[0] 进行截取，确保其在 [2.6, 4] 范围内
        action[1] = np.random.normal(loc=0, scale=0.4)  # 使用 numpy 生成在指定均值和方差下的随机数作为 action[0]
        action[1] = np.clip(action[1], -1.5, 1.5)  # 按照 [-1.2, 1.2] 的区间截取 action[1]（横向偏移）
        action[2] = np.random.normal(loc=0, scale=1.2)  # 使用 numpy 生成在指定均值和方差下的随机数作为 action[0]        
        action[2] = np.clip(action[2], -3.14, 3.14)  # 按照 [-3.14, 3.14] 的区间截取 action[2]（初始角速度）

        return action

#创建Critic网络类继承自nn.Module
class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(80, 128)       # 定义全连接层1
        self.fc2 = nn.Linear(128, 64)       # 定义全连接层2
        self.fc3 = nn.Linear(64, 10)        # 定义全连接层3
        self.out = nn.Linear(10, 1)         # 定义输出层
        self.out.weight.data.mul_(0.1)      # 初始化输出层权重

    def forward(self, x):
        x = self.fc1(x)                     # 输入张量经全连接层1传递
        x = torch.tanh(x)                   # 经tanh函数激活
        x = self.fc2(x)                     # 经全连接层2传递
        x = torch.sigmoid(x)                   # 经tanh函数激活
        x = self.fc3(x)                     # 经全连接层3传递
        x = torch.tanh(x)                   # 经tanh函数激活
        return self.out(x)                  # 经输出层传递得到输出张量
    
BATCH_SIZE = 32                             # 批次尺寸
GAMMA = 0.92                                 # 奖励折扣因子
LAMDA = 0.88                                 # GAE算法的调整因子
EPSILON = 0.1                               # 截断调整因子

#生成动态学习率
def LearningRate(x):
    lr_start = 0.0001                       # 起始学习率
    lr_end = 0.0005                         # 终止学习率
    lr_decay = 20000                        # 学习率衰减因子
    return lr_end + (lr_start - lr_end) * math.exp(-1. * x / lr_decay)

# 输出连续动作的概率分布
def log_density(x, mu, std, logstd):
    var = std.pow(2)
    log_density = -(x - mu).pow(2) / (2 * var) - 0.5 * math.log(2 * math.pi) - logstd
    return log_density.sum(1, keepdim=True)

# 使用GAE方法计算优势函数
def get_gae(rewards, masks, values):
    rewards = torch.Tensor(rewards)
    masks = torch.Tensor(masks)
    returns = torch.zeros_like(rewards)
    advants = torch.zeros_like(rewards)
    running_returns = 0
    previous_value = 0
    running_advants = 0

    for t in reversed(range(0, len(rewards))):
        running_returns = rewards[t] + GAMMA * running_returns * masks[t]
        running_tderror = rewards[t] + GAMMA * previous_value * masks[t] - values.data[t]
        running_advants = running_tderror + GAMMA * LAMDA * running_advants * masks[t]

        returns[t] = running_returns
        previous_value = values.data[t]
        advants[t] = running_advants
    advants = (advants - advants.mean()) / advants.std()
    return returns, advants

# 替代损失函数
def surrogate_loss(actor, advants, states, old_policy, actions, index):
    mu, std, logstd = actor(torch.Tensor(states))
    new_policy = log_density(actions, mu, std, logstd)
    old_policy = old_policy[index]
    ratio = torch.exp(new_policy - old_policy)
    surrogate = ratio * advants
    return surrogate, ratio

# 训练模型
def train_model(actor, critic, memory, actor_optim, critic_optim):
    memory = np.array(memory, dtype=object)
    states = np.vstack(memory[:, 0])
    actions = list(memory[:, 1])
    rewards = list(memory[:, 2])
    masks = list(memory[:, 3])
    values = critic(torch.Tensor(states))
    loss_list = []
    # step 1: get returns and GAEs and log probability of old policy
    returns, advants = get_gae(rewards, masks, values)
    mu, std, logstd = actor(torch.Tensor(states))
    old_policy = log_density(torch.Tensor(np.array(actions)), mu, std, logstd)
    old_values = critic(torch.Tensor(states))
    criterion = torch.nn.MSELoss()
    n = len(states)
    arr = np.arange(n)
    # step 2: get value loss and actor loss and update actor & critic
    for epoch in range(10):
        np.random.shuffle(arr)
        for i in range(n // BATCH_SIZE):
            batch_index = arr[BATCH_SIZE * i: BATCH_SIZE * (i + 1)]
            batch_index = torch.LongTensor(batch_index)
            inputs = torch.Tensor(states)[batch_index]
            returns_samples = returns.unsqueeze(1)[batch_index]
            advants_samples = advants.unsqueeze(1)[batch_index]
            actions_samples = torch.Tensor(np.array(actions))[batch_index]
            oldvalue_samples = old_values[batch_index].detach()
            loss, ratio = surrogate_loss(actor, advants_samples, inputs,
                                         old_policy.detach(), actions_samples,
                                         batch_index)
            values = critic(inputs)
            clipped_values = oldvalue_samples + torch.clamp(values - oldvalue_samples, -EPSILON, EPSILON)
            critic_loss1 = criterion(clipped_values, returns_samples)
            critic_loss2 = criterion(values, returns_samples)
            critic_loss = torch.max(critic_loss1, critic_loss2).mean()

            clipped_ratio = torch.clamp(ratio, 1.0 - EPSILON, 1.0 + EPSILON)
            clipped_loss = clipped_ratio * advants_samples
            actor_loss = -torch.min(loss, clipped_loss).mean()

            loss = actor_loss + critic_loss

            loss_list.append(loss)
            critic_optim.zero_grad()
            critic_loss.backward(retain_graph=True)
            critic_optim.step()

            actor_optim.zero_grad()
            actor_loss.backward()
            actor_optim.step()

    return 0, sum(loss_list)/10
def calculate_counts(init_score, infostate):
    hit_count = 0
    placeholder_count = 0
    protect_count = 0

    for i in range(0, len(infostate), 5):
        x = infostate[i]
        y = infostate[i + 1]
        valid = infostate[i + 4]

        if valid == 1:
            sn = infostate[i + 3]
            if sn == 1:  # 当前投掷顺序为先手
                protect = False
                for j in range(0, len(infostate), 5):
                    other_x = infostate[j]
                    other_y = infostate[j + 1]
                    other_valid = infostate[j + 4]
                    if other_valid == 1 and j != i:
                        if other_x == x and other_y < y:
                            protect = True
                            break
                if protect:
                    protect_count += 1
                else:
                    hit_count += 1
            else:  # 当前投掷顺序为后手
                hit_count += 1
        else:
            placeholder_count += 1

    return hit_count, placeholder_count, protect_count

import time, os
from kun import AIRobot
import argparse

class PPORobot(AIRobot):
    def __init__(self, key, name, host, port, round_max=10000):
        super().__init__(key=key, name=name, host=host, port=port)

        #初始化并加载先手actor模型
        self.init_actor = Actor()
        self.init_actor_file = 'model/PPO_init_actor.pth'
        if os.path.exists(self.init_actor_file):
            print("加载模型文件 %s" % (self.init_actor_file))
            self.init_actor.load_state_dict(torch.load(self.init_actor_file))

        #初始化并加载先手critic模型
        self.init_critic = Critic()
        self.init_critic_file = 'model/PPO_init_critic.pth'
        if os.path.exists(self.init_critic_file):
            print("加载模型文件 %s" % (self.init_critic_file))
            self.init_critic.load_state_dict(torch.load(self.init_critic_file))

        #初始化并加载后手actor模型
        self.dote_actor = Actor()
        self.dote_actor_file = 'model/PPO_dote_actor.pth'
        if os.path.exists(self.dote_actor_file):
            print("加载模型文件 %s" % (self.dote_actor_file))
            self.dote_actor.load_state_dict(torch.load(self.dote_actor_file))
  
        #初始化并加载后手critic模型
        self.dote_critic = Critic()        
        self.dote_critic_file = 'model/PPO_dote_critic.pth'
        if os.path.exists(self.dote_critic_file):
            print("加载模型文件 %s" % (self.dote_critic_file))
            self.dote_critic.load_state_dict(torch.load(self.dote_critic_file))
          
        self.memory = deque()               # 清空经验数据
        self.round_max = round_max          # 最多训练局数
        self.log_file_name = 'log/PPO_log/traindata_' + time.strftime("%y%m%d_%H%M%S") + '.log' # 日志文件     

    #根据当前比分获取奖励分数
    def get_reward(self, this_score, hit_count, placeholder_count, protect_count):
        House_R = 1.830
        Stone_R = 0.145
        reward = this_score - self.last_score

        # 如果得分相同，根据投掷位置距离营垒的远近以及击中或击飞的冰壶数量给予奖励或惩罚
        if reward == 0:
            x = self.position[2 * self.shot_num]
            y = self.position[2 * self.shot_num + 1]
            dist = self.get_dist(x, y)
            
            if dist < (House_R + Stone_R):
                # 根据距离营垒的远近给予奖励或惩罚
                reward = 1 - dist / (House_R + Stone_R)
                if this_score > 0:
                    reward *= 1  # 奖励的比例可以根据具体需求进行调整
                else:
                    reward *= -1  # 惩罚的比例可以根据具体需求进行调整
            if y>8.8 or y<2:
                reward=reward-0.5
            # 根据击中或击飞的冰壶数量给予奖励
            hit_reward = hit_count * 1.2  # 每个击中或击飞的冰壶奖励0.2分，可以根据具体需求进行调整
            reward += hit_reward

            # 根据占位球和保护冰壶的情况给予奖励
            placeholder_reward = placeholder_count * 0.3  # 每个占位球奖励0.1分，可以根据具体需求进行调整
            protect_reward = protect_count * 0.3  # 每个保护冰壶奖励0.1分，可以根据具体需求进行调整
            reward += placeholder_reward + protect_reward

        return reward
        
    #处理投掷状态消息
    def recv_setstate(self, msg_list):
        #当前完成投掷数
        self.shot_num = int(msg_list[0])
        #总对局数
        self.round_total = int(msg_list[2])

        #达到最大局数则退出训练
        if self.round_num == self.round_max:
            self.on_line = False
            return
        
        #每一局开始时将历史比分清零
        if (self.shot_num == 0):
            self.last_score = 0
        this_score = 0
            
        #根据先后手选取模型并设定当前选手第一壶是当局比赛的第几壶
        if self.player_is_init:
            first_shot = 0
            self.actor = self.init_actor
            self.critic = self.init_critic
        else:
            first_shot = 1
            self.actor = self.dote_actor
            self.critic = self.dote_critic
            
        #当前选手第1壶投出前
        if self.shot_num == first_shot:
            init_score, self.s1 = get_infostate(self.position)      # 获取当前得分和状态描述
            self.action = self.actor.choose_action(self.s1)         # 根据状态获取对应的动作参数列表
            self.last_score = (1-2*first_shot)*init_score           # 先手为正/后手为负
        #当前选手第1壶投出后
        if self.shot_num == first_shot+1:
            init_score, _ = get_infostate(self.position)            # 获取当前得分和状态描述
            this_score = (1-2*first_shot)*init_score                # 先手为正/后手为负
       # 先手为正/后手为负
            hit, placeholder, protect = calculate_counts(init_score, _)
            reward = self.get_reward(this_score, hit, placeholder, protect)  # 获取动作奖励
            self.memory.append([self.s1, self.action, reward, 1])   # 保存经验数据
        #当前选手第2壶投出前
        if self.shot_num == first_shot+2:
            init_score, self.s2 = get_infostate(self.position)      # 获取当前得分和状态描述
            self.action = self.actor.choose_action(self.s2)         # 根据状态获取对应的动作参数列表
            self.last_score = (1-2*first_shot)*init_score           # 先手为正/后手为负
        #当前选手第2壶投出后
        if self.shot_num == first_shot+3:
            init_score, _ = get_infostate(self.position)            # 获取当前得分和状态描述
            this_score = (1-2*first_shot)*init_score                # 先手为正/后手为负
       # 先手为正/后手为负
            hit, placeholder, protect = calculate_counts(init_score, _)
            reward = self.get_reward(this_score, hit, placeholder, protect)  # 获取动作奖励
            self.memory.append([self.s2, self.action, reward, 1])   # 保存经验数据
        #当前选手第3壶投出前
        if self.shot_num == first_shot+4:
            init_score, self.s3 = get_infostate(self.position)      # 获取当前得分和状态描述
            self.action = self.actor.choose_action(self.s3)         # 根据状态获取对应的动作参数列表
            self.last_score = (1-2*first_shot)*init_score           # 先手为正/后手为负
        #当前选手第3壶投出后
        if self.shot_num == first_shot+5:
            init_score, _ = get_infostate(self.position)            # 获取当前得分和状态描述
            this_score = (1-2*first_shot)*init_score                # 先手为正/后手为负
       # 先手为正/后手为负
            hit, placeholder, protect = calculate_counts(init_score, _)
            reward = self.get_reward(this_score, hit, placeholder, protect)  # 获取动作奖励
            self.memory.append([self.s3, self.action, reward, 1])   # 保存经验数据
        #当前选手第4壶投出前
        if self.shot_num == first_shot+6:
            init_score, self.s4 = get_infostate(self.position)      # 获取当前得分和状态描述
            self.action = self.actor.choose_action(self.s4)         # 根据状态获取对应的动作参数列表
            self.last_score = (1-2*first_shot)*init_score           # 先手为正/后手为负
        #当前选手第4壶投出后
        if self.shot_num == first_shot+7:
            init_score, _ = get_infostate(self.position)            # 获取当前得分和状态描述
            this_score = (1-2*first_shot)*init_score                # 先手为正/后手为负
       # 先手为正/后手为负
            hit, placeholder, protect = calculate_counts(init_score, _)
            reward = self.get_reward(this_score, hit, placeholder, protect)  # 获取动作奖励
            self.memory.append([self.s4, self.action, reward, 1])   # 保存经验数据
        #当前选手第5壶投出前
        if self.shot_num == first_shot+8:
            init_score, self.s5 = get_infostate(self.position)      # 获取当前得分和状态描述
            self.action = self.actor.choose_action(self.s5)         # 根据状态获取对应的动作参数列表
            self.last_score = (1-2*first_shot)*init_score           # 先手为正/后手为负
        #当前选手第5壶投出后
        if self.shot_num == first_shot+9:
            init_score, _ = get_infostate(self.position)            # 获取当前得分和状态描述
            this_score = (1-2*first_shot)*init_score                # 先手为正/后手为负
       # 先手为正/后手为负
            hit, placeholder, protect = calculate_counts(init_score, _)
            reward = self.get_reward(this_score, hit, placeholder, protect)  # 获取动作奖励
            self.memory.append([self.s5, self.action, reward, 1])   # 保存经验数据
        #当前选手第6壶投出前
        if self.shot_num == first_shot+10:
            init_score, self.s6 = get_infostate(self.position)      # 获取当前得分和状态描述
            self.action = self.actor.choose_action(self.s6)         # 根据状态获取对应的动作参数列表
            self.last_score = (1-2*first_shot)*init_score           # 先手为正/后手为负
        #当前选手第6壶投出后
        if self.shot_num == first_shot+11:
            init_score, _ = get_infostate(self.position)            # 获取当前得分和状态描述
            this_score = (1-2*first_shot)*init_score                # 先手为正/后手为负
       # 先手为正/后手为负
            hit, placeholder, protect = calculate_counts(init_score, _)
            reward = self.get_reward(this_score, hit, placeholder, protect)  # 获取动作奖励
            self.memory.append([self.s6, self.action, reward, 1])   # 保存经验数据
        #当前选手第7壶投出前
        if self.shot_num == first_shot+12:
            init_score, self.s7 = get_infostate(self.position)      # 获取当前得分和状态描述
            self.action = self.actor.choose_action(self.s7)         # 根据状态获取对应的动作参数列表
            self.last_score = (1-2*first_shot)*init_score           # 先手为正/后手为负
        #当前选手第7壶投出后
        if self.shot_num == first_shot+13:
            init_score, _ = get_infostate(self.position)            # 获取当前得分和状态描述
            this_score = (1-2*first_shot)*init_score                # 先手为正/后手为负
       # 先手为正/后手为负
            hit, placeholder, protect = calculate_counts(init_score, _)
            reward = self.get_reward(this_score, hit, placeholder, protect)  # 获取动作奖励
            self.memory.append([self.s7, self.action, reward, 1])   # 保存经验数据
        #当前选手第8壶投出前
        if self.shot_num == first_shot+14:
            init_score, self.s8 = get_infostate(self.position)      # 获取当前得分和状态描述
            self.action = self.actor.choose_action(self.s8)         # 根据状态获取对应的动作参数列表
            
        if self.shot_num == 16:
            if self.score > 0:
                reward = 10 * self.score                            # 获取动作奖励
            else:
                reward = 0      
            self.memory.append([self.s8, self.action, reward, 0])   # 保存经验数据
            
            self.round_num += 1
            #如果处于训练模式且有12局数据待训练
            if (self.round_max > 0) and (self.round_num % 12 == 0):
                #训练模型
                actor_optim = torch.optim.Adam(self.actor.parameters(), lr=LearningRate(self.round_num))
                critic_optim = torch.optim.Adam(self.critic.parameters(), lr=LearningRate(self.round_num),
                                                weight_decay=0.0001)
                self.actor.train(), self.critic.train()
                _, loss = train_model(self.actor, self.critic, self.memory, actor_optim, critic_optim)
                #保存模型
                if self.player_is_init:
                    torch.save(self.actor.state_dict(), self.init_actor_file)
                    torch.save(self.critic.state_dict(), self.init_critic_file)
                else:
                    torch.save(self.actor.state_dict(), self.dote_actor_file)
                    torch.save(self.critic.state_dict(), self.dote_critic_file)
                print('============= Checkpoint Saved =============')
                #清空训练数据
                self.memory = deque()
                
            #将本局比分和当前loss值写入日志文件
            #log_file = open(self.log_file_name, 'a+')
            #log_file.write("score "+str(self.score)+" "+str(self.round_num)+"\n")
            #if self.round_num % 12 == 0:
                #log_file.write("loss "+str(float(loss))+" "+str(self.round_num)+"\n")
            #log_file.close()
            
    def get_bestshot(self):
        return  "BESTSHOT " + str(self.action)[1:-1].replace(',', '')




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='KunBot')
    parser.add_argument('-H','--host', help='tcp server host', default='192.168.5.76', required=False)
    parser.add_argument('-p','--port', help='tcp server port', default=7788, required=False)
    args, unknown = parser.parse_known_args()
    print(args)

    #根据数字冰壶服务器界面中给出的连接信息修改CONNECTKEY，注意这个数据每次启动都会改变。
    key = "lidandan_0fd99bfd-8fd7-4d36-8250-3d83f1367b0"
    #初始化AI选手
    airobot = PPORobot(key=key, name="KunBot", host=args.host, port=int(args.port))
    #启动AI选手处理和服务器的通讯
    airobot.recv_forever()
    #导入matplotlib函数库