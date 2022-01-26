import cv2
import psutil
import numpy as np


def memory_usage():
    p = psutil.Process()
    return p.memory_info().rss / 2 ** 20 # Bytes to MiB

def atari_preprocessing(observation: np.array):
    frame = cv2.cvtColor(observation, cv2.COLOR_BGR2GRAY)
    resized_frame = cv2.resize(frame, (84, 110), interpolation = cv2.INTER_LINEAR)
    return np.reshape(resized_frame[110-84:110, 0:84], (1,84,84))

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