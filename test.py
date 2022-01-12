from ale_py import ALEInterface
from ale_py.roms import Breakout
import gym
import time
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from Model.dqn import DQN
import torch
import torch.optim as optim
import random
import sys

def atari_preprocessing(state):
    state_img = Image.fromarray(state).convert("L")
    state_img = state_img.resize((84, 110))
    state_img = state_img.crop((0, 110-84, 84, 110))
    return np.array(state_img)/255

def train_minibatch(model: DQN, minibatch, optimizer: optim.RMSprop, gamma: float):
    states = list(map(lambda x: x[0], minibatch))
    actions = list(map(lambda x: x[1], minibatch))
    rewards = list(map(lambda x: x[2], minibatch))
    next_states = list(map(lambda x: x[3], minibatch))
    dones = list(map(lambda x: x[4], minibatch))

    optimizer.zero_grad()
    q_values = list(map(lambda state, action: model(state.cuda())[0, action], states, actions))
    q_values_stack = torch.stack(q_values, dim=0)

    ys = list(map(lambda reward, next_state, done: torch.tensor(reward).cuda() if done else torch.tensor(reward).cuda() + gamma*torch.max(model(next_state.cuda())), rewards, next_states, dones))
    ys_stack = torch.stack(ys, dim=0)

    loss = torch.mean(torch.pow(ys_stack-q_values_stack, 2))
    loss.backward()
    optimizer.step()
    return model

ale = ALEInterface()
ale.loadROM(Breakout)


# env = gym.make('ALE/Breakout-v5')
env = gym.make('Breakout-v0')

env.reset()

print(env.action_space)
print(env.observation_space.dtype, env.observation_space._shape)
# print(env.get_keys_to_action())


max_timestep = 10000000
max_episode = 1000
replay_memory = []
capacity = 1000000
epsilon = 1
epsilon_bound = 0.1
epsilon_eval = 0.05
epsilon_degrade = (1-0.1)/1000000
minibatch_size = 32
timestep = 1
render = False
time_interval = 0.01
learning_rate = 0.001
gamma = 0.99
n_action = env.action_space.n

model = DQN(n_action)

optimizer = optim.RMSprop(model.parameters(), lr=learning_rate)
eval_results = []


for episode in range(max_episode):
    ''' Training Phase '''
    model.train()
    observation = env.reset()
    reward_sum = 0

    if render:
        env.render()
        time.sleep(time_interval)


    gray = atari_preprocessing(observation)
    gray_torch = torch.Tensor(gray)
    grays = [gray_torch]*4
    state = torch.stack(grays, dim=0).unsqueeze(0)

    while True:
        ''' action selection '''
        if random.random() < epsilon: 
            action = random.randint(0, n_action-1)
        else:
            if torch.cuda.is_available():
                state = state.cuda()
            output = model(state)
            action = torch.argmax(output).item()


        ''' Skip frame action '''
        for _ in range(4):
            observation, reward, done, info = env.step(action)
            reward_sum += reward
            if render:
                env.render()
                time.sleep(time_interval)

            if epsilon_bound < epsilon:
                epsilon = epsilon - epsilon_degrade


            gray = atari_preprocessing(observation)
            gray_torch = torch.Tensor(gray)
            grays = grays[1:] + [gray_torch]
            next_state = torch.stack(grays, dim=0).unsqueeze(0)
            
            if done:
                reward = -1.0
            
            replay_memory.append((state, action, reward, next_state, done))
            if len(replay_memory) > capacity:
                replay_memory.pop(0)
            state = next_state
        if len(replay_memory) >= minibatch_size:
            minibatch = random.sample(replay_memory, minibatch_size)
            model = train_minibatch(model=model, minibatch=minibatch, optimizer=optimizer, gamma=gamma)        
    
        if done:
            break

    ''' Validation Phase '''
    model.eval()
    observation = env.reset()
    reward_sum = 0

    if render:
        env.render()
        time.sleep(time_interval)


    gray = atari_preprocessing(observation)
    gray_torch = torch.Tensor(gray)
    grays = [gray_torch]*4
    eval_state = torch.stack(grays, dim=0).unsqueeze(0)


    while True:
        ''' action selection '''
        if random.random() < epsilon_eval: 
            action = random.randint(0, n_action-1)
        else:
            if torch.cuda.is_available():
                eval_state = eval_state.cuda()
            output = model(eval_state)
            action = torch.argmax(output).item()

        observation, reward, done, info = env.step(action)
        reward_sum += reward
        if render:
            env.render()
            time.sleep(time_interval)

        gray = atari_preprocessing(observation)
        gray_torch = torch.Tensor(gray)
        grays = grays[1:] + [gray_torch]
        next_eval_state = torch.stack(grays, dim=0).unsqueeze(0)
        eval_state = next_eval_state

        if done:
            break

    print(f"[{episode}] reward_sum: {reward_sum}, epsilon: {epsilon}")
    eval_results.append(reward_sum)


plt.plot(eval_results)
plt.show()
