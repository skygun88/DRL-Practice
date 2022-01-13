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
max_epoch = 5
epsilon_eval = 0.05
eval_timestep = 0
eval_max_timestep = 1000
render = False
time_interval = 0.01
learning_rate = 0.001
gamma = 0.99
n_action = env.action_space.n - 1
history = [] # state - 4 skip frames
images = []


action_map = {0:0, 1:2, 2:3}
model = DQN(n_action)
model.load_state_dict(torch.load(f'Weight/DQN_breakout_30.pt'))
model.eval()

video = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 60, (160, 210))

eval_results = []

for epoch in range(max_epoch):
    model.eval()
    observation = env.reset()
    history.clear()
    reward_sum = 0
    prev_lives = 5
    eval_timestep = 0
    epi_rewards = []
    dead = False

    if render:
        env.render()
        time.sleep(time_interval)
    
    while True:
        # images.append(observation)
        
        video.write(cv2.cvtColor(observation, cv2.COLOR_BGR2RGB))
        if len(history) < 4:
            if len(history) < 1:
                observation, _, _, _ = env.step(1) # Initial Fire
            else:
                observation, _, _, _ = env.step(0)
            frame = atari_preprocessing(observation)
            history.append(frame)
            continue
        
        state = torch.stack(history, dim=0).unsqueeze(0)

        if dead:
            observation, _, _, _ = env.step(1) # Initial Fire
            frame = atari_preprocessing(observation)
            history.pop(0)
            history.append(frame)
            dead = False
            continue

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

        

        if done:
            history.clear()
            env.reset()
            prev_lives = 5
            epi_rewards.append(reward_sum)
            reward_sum = 0
            continue

    print(f"[Evaluate] Epoch: {epoch}, episode reward: {sum(epi_rewards)/len(epi_rewards)}")


video.release()