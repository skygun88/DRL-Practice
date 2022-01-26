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
from utils import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate():
    env = gym.make('BreakoutDeterministic-v4')
    env.reset()

    max_epoch = 1
    epsilon_eval = 0.01
    eval_max_timestep = 10000
    render = False
    time_interval = 0.01
    # n_action = env.action_space.n - 1
    # action_map = {0:0, 1:2, 2:3}

    n_action = env.action_space.n 
    action_map = {0:0, 1:1, 2:2, 3:3}

    model = DQN(n_action)
    model.load_state_dict(torch.load(f'Weight/DQN_breakout_99.pt'))
    model.eval()

    video = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (160, 210))

    for epoch in range(max_epoch):
        model.eval()
        observation = env.reset()
        reward_sum = 0
        prev_lives = 5
        epi_rewards = []
        dead = False
        q_value_sum = 0
        # rewards = []

        observation, _, _, _ = env.step(1)
        frame = atari_preprocessing(observation)
        history = np.concatenate((frame,frame,frame,frame), axis=0)    

        if render:
            env.render()
            time.sleep(time_interval)
        
        for _ in range(eval_max_timestep):
            video.write(cv2.cvtColor(observation, cv2.COLOR_BGR2RGB))
            state = torch.Tensor(history/255).unsqueeze(0).to(device=device)

            ''' action selection '''
            with torch.no_grad():
                output = model(state)
                if random.random() < epsilon_eval: 
                    action = random.randint(0, n_action-1)
                else:
                    action = torch.argmax(output).item()
                q_value = torch.amax(output).item()
            real_action = action_map[action]
            
            q_value_sum += q_value

            ''' Environment update '''
            observation, reward, done, info = env.step(real_action)
            if render:
                env.render()
                time.sleep(time_interval)
            
            dead = is_dead(info, prev_lives=prev_lives)
            reward_sum += reward
            # rewards.append(reward)

            ''' Next state '''
            frame = atari_preprocessing(observation)
            history = np.append(history[1:, :, :], frame, axis=0)

            ''' Training parameter update '''
            prev_lives = info['lives']

            if dead:
                observation, _, _, _ = env.step(1) # Initial Fire
                frame = atari_preprocessing(observation)
                history = np.append(history[1:, :, :], frame, axis=0)
                dead = False
                state = torch.Tensor(history/255).unsqueeze(0).to(device=device)

            if done:
                env.reset()
                observation, _, _, _ = env.step(1)
                frame = atari_preprocessing(observation)
                history = np.concatenate((frame,frame,frame,frame), axis=0)
                state = torch.Tensor(history/255).unsqueeze(0).to(device=device)
                prev_lives = 5
                epi_rewards.append(reward_sum)
                reward_sum = 0
                continue


        avg_epi_rewards = sum(epi_rewards)/len(epi_rewards)
        avg_q_values = q_value_sum/eval_max_timestep
        print(f"[Validate] Epoch: {epoch}, episode reward: {avg_epi_rewards:.2f}, max q_value: {avg_q_values:.3f}")
        # print(list(filter(lambda x: x > 0, rewards)))


    video.release()

if __name__ == '__main__':
    evaluate()