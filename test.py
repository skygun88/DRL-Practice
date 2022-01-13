
import sys
import cv2
import gym
import time
import random
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
import torch.nn.functional as F
from ale_py import ALEInterface
from ale_py.roms import Breakout
from PIL import Image
from Model.dqn import DQN


def atari_preprocessing(state):
    state_img = Image.fromarray(state).convert("L")
    state_img = state_img.resize((84, 110))
    state_img = state_img.crop((0, 110-84, 84, 110))
    return torch.Tensor(np.array(state_img)/255)

def train_minibatch(model: DQN, minibatch, optimizer: optim.RMSprop, gamma: float):
    optimizer.zero_grad()
    states = torch.stack(list(map(lambda x: x[0], minibatch))).squeeze()
    actions = list(map(lambda x: x[1], minibatch))
    one_hot_actions = F.one_hot(torch.tensor(actions), num_classes=3)
    rewards = torch.tensor(list(map(lambda x: x[2], minibatch)))
    next_states = torch.stack(list(map(lambda x: x[3], minibatch))).squeeze()
    dones = torch.tensor(list(map(lambda x: 1 if x[4] == True or x[2] < 0 else 0, minibatch)))

    q_values = torch.sum(model(states.cuda())*one_hot_actions.cuda(), 1)
    ys = rewards.cuda() + (1-dones.cuda())*gamma*torch.amax(model(next_states.cuda()), 1)

    loss = torch.mean(torch.pow(ys-q_values, 2))
    loss.backward()
    optimizer.step()
    return model

# def train_minibatch(model: DQN, minibatch, optimizer: optim.RMSprop, gamma: float):
#     states = list(map(lambda x: x[0], minibatch))
#     actions = list(map(lambda x: x[1], minibatch))
#     rewards = list(map(lambda x: x[2], minibatch))
#     next_states = list(map(lambda x: x[3], minibatch))
#     dones = list(map(lambda x: x[4], minibatch))

#     optimizer.zero_grad()
#     q_values = list(map(lambda state, action: model(state.cuda())[0, action], states, actions))
#     q_values_stack = torch.stack(q_values, dim=0)

#     ys = list(map(lambda reward, next_state, done: torch.tensor(reward).cuda() if done or reward < 0 else torch.tensor(reward).cuda() + gamma*torch.max(model(next_state.cuda())), rewards, next_states, dones))
#     ys_stack = torch.stack(ys, dim=0)

#     loss = torch.mean(torch.pow(ys_stack-q_values_stack, 2))
#     loss.backward()
#     optimizer.step()
#     return model

def reward_processing(reward, dead):
    if reward > 0:
        return 1
    if dead:
        return -1
    return 0

def is_dead(info, prev_lives):
    if info['lives'] < prev_lives:
        return True
    return False

env = gym.make('BreakoutDeterministic-v4')
env.reset()

print(env.action_space)
print(env.observation_space.dtype, env.observation_space._shape)

max_timestep = 10000000
max_epoch = 100
replay_memory = []
capacity = 50000
epsilon = 1
epsilon_bound = 0.1
epsilon_eval = 0.05
epsilon_degrade = (1-0.1)/1000000
minibatch_size = 32
minibatch_train = 50000 # The number of minibatches trained an epoch
minibatch_cnt = 0
train_start = 10000
timestep = 0
eval_timestep = 0
eval_max_timestep = 10000
render = False
time_interval = 0.01
learning_rate = 0.001
gamma = 0.99
n_action = env.action_space.n - 1
history = [] # state - 4 skip frames


action_map = {0:0, 1:2, 2:3}
model = DQN(n_action)

optimizer = optim.RMSprop(model.parameters(), lr=learning_rate)
eval_results = []


for epoch in range(max_epoch):
    ''' Training Phase '''
    model.train()
    env.reset()
    history.clear()
    reward_sum = 0
    prev_lives = 5
    timestep = 0
    minibatch_cnt = 0

    if render:
        env.render()
        time.sleep(time_interval)

    while True:
        if len(history) < 4:
            if len(history) < 1:
                observation, _, _, _ = env.step(1) # Initial Fire
            else:
                observation, _, _, _ = env.step(0)
            frame = atari_preprocessing(observation)
            history.append(frame)
            continue
        
        state = torch.stack(history, dim=0).unsqueeze(0)
        
        ''' action selection '''
        if random.random() < epsilon: 
            action = random.randint(0, n_action-1)
        else:
            if torch.cuda.is_available():
                output = model(state.cuda())
            else:
                output = model(state)
            action = torch.argmax(output).item()
        real_action = action_map[action]


        ''' Environment update '''
        observation, reward, done, info = env.step(real_action)
        if render:
            env.render()
            time.sleep(time_interval)
        
        dead = is_dead(info, prev_lives=prev_lives)
        reward = reward_processing(reward, dead)
        reward_sum += reward

        ''' Next state '''
        frame = atari_preprocessing(observation)
        history.pop(0)
        history.append(frame)
        next_state = torch.stack(history, dim=0).unsqueeze(0)
        
        ''' Replay memory update '''
        replay_memory.append((state, action, reward, next_state, done))
        if len(replay_memory) > capacity:
            replay_memory.pop(0)
        
        if len(replay_memory) > train_start:
            minibatch = random.sample(replay_memory, minibatch_size)
            model = train_minibatch(model=model, minibatch=minibatch, optimizer=optimizer, gamma=gamma)
            minibatch_cnt += 1

        ''' Training parameter update '''
        prev_lives = info['lives']
        timestep += 1

        if timestep % 1000 == 0:
            print(f'[Training] Epoch: {epoch}, Timestep: {timestep}, reward_sum = {reward_sum}')

        if epsilon_bound < epsilon:
            epsilon = max(epsilon - epsilon_degrade, epsilon_bound)
        
        if minibatch_cnt == minibatch_train:
            break

        if dead:
            history.clear()

        if done:
            history.clear()
            env.reset()
            prev_lives = 5
            continue

    torch.save(model.state_dict(), f'Weight/DQN_breakout_{epoch}.pt')

    ''' Validation Phase '''
    model.eval()
    env.reset()
    history.clear()
    reward_sum = 0
    prev_lives = 5
    eval_timestep = 0
    epi_rewards = []

    if render:
        env.render()
        time.sleep(time_interval)
    
    while True:
        if len(history) < 4:
            if len(history) < 1:
                observation, _, _, _ = env.step(1) # Initial Fire
            else:
                observation, _, _, _ = env.step(0)
            frame = atari_preprocessing(observation)
            history.append(frame)
            continue
        
        state = torch.stack(history, dim=0).unsqueeze(0)

        ''' action selection '''
        if random.random() < epsilon_eval: 
            action = random.randint(0, n_action-1)
        else: 
            if torch.cuda.is_available():
                output = model(state.cuda())
            else:
                output = model(state)
            action = torch.argmax(output).item()
        real_action = action_map[action]


        ''' Environment update '''
        observation, reward, done, info = env.step(real_action)
        if render:
            env.render()
            time.sleep(time_interval)
        
        dead = is_dead(info, prev_lives=prev_lives)
        reward = reward_processing(reward, dead)
        reward_sum += reward

        ''' Next state '''
        frame = atari_preprocessing(observation)
        history.pop(0)
        history.append(frame)

        ''' Training parameter update '''
        prev_lives = info['lives']
        eval_timestep += 1

        if eval_timestep >= eval_max_timestep:
            break     

        if dead:
            history.clear()

        if done:
            history.clear()
            env.reset()
            prev_lives = 5
            epi_rewards.append(reward_sum)
            reward_sum = 0
            continue

    print(f"[Validation] Epoch: {epoch}, episode reward: {sum(epi_rewards)/len(epi_rewards)}, epsilon: {epsilon}")
    eval_results.append(reward_sum)


plt.plot(eval_results)
plt.show()
