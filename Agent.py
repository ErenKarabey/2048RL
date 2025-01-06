# -*- coding: utf-8 -*-
"""
Created on Mon Jan  6 08:17:49 2025

@author: karab
"""

import sys
# You need to change here
newpath = 'C:/Users/karab/Downloads/2048reinf/2048Reinf'
if newpath not in sys.path: sys.path.append(newpath)

import Board
import Visuals

import numpy as np
import random
import time

import numpy as np
import matplotlib.pyplot as plt

import pygame
import cv2
            
class TDAfterStateAgent:
    def __init__(self, env, alpha=0.1, gamma=0.9, epsilon=0.1, learning_enabled=True):
        self.env = env
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate
        self.learning_enabled = learning_enabled  # Learning flag
        
        self.V = {}  # State values dictionary
        
        self.training_info = [] #recording score and max tile for every game
        self.play_freqs = []
        
    def get_state_value(self, state):
        """Returns the value of a state (0 if not seen)."""
        return self.V.get(tuple(state.flatten()), 0)
    
    def td_update(self, state, action, reward, next_state):
        """Perform TD(0) update."""
        state_value = self.get_state_value(state)
        next_state_value = self.get_state_value(next_state)
        
        # TD(0) update rule: V(s) <- V(s) + alpha * [reward + gamma * V(s') - V(s)]
        self.V[tuple(state.flatten())] = state_value + self.alpha * (reward + self.gamma * next_state_value - state_value)
    
    def choose_action(self, state):
        """Choose an action using epsilon-greedy policy."""
        if np.random.rand() < self.epsilon:
            return self.env.action_space.sample()  # Random action (exploration)
        else:
            state_values = [-1, -1, -1, -1]
            next_states = self.env.get_next_states()
            for (action, reward, next_state) in next_states:
                state_values[action] = self.get_state_value(next_state)
            return np.argmax(state_values)  # Choose the action with the highest state value
    
    def all_next_states(self, state):
        iss, jss = np.where(state == 0)
        
        states = []
        for i, j in zip(iss, jss):
            _state = np.copy(state)
            _state[i, j] = 2
            states.append((0.9, _state))
            _state[i, j] = 4
            states.append((0.1, _state))
        return states    
    
    def evaluate(self, state, action):
        next_state, reward, done, score = self.env.step(action, display=False, spawn=False)
        next_states = self.all_next_states(next_state)
        
        return reward + np.sum([p * self.get_state_value(s) for p, s in next_states])
    
    def make_move(self, state, action, display=False):
        """Computes afterstate after making a move."""
        next_state, reward, done, score = self.env.step(action, display=display, spawn=True)
        next_states = self.all_next_states(next_state)
        return reward, done, next_state, next_states
    
    def set_learning(self, learn):
        self.learning_enabled = learn
    
    def policy(self, state):
        return max(range(self.env.action_space.n), key=lambda a: self.evaluate(state, a))
    
    def play_game(self, test=False):
        """Play a game following the pseudocode."""
        score = 0
        state = self.env.reset()
        done = False
        
        display = not self.learning_enabled and not test
        while not done:
            # Choose the best action (arg max of evaluate)
            action = self.policy(state)
            
            reward, done, next_state, next_states = self.make_move(state, action, display=display)
            
            if self.learning_enabled:
                self.td_update(state, action, reward, next_state)
            
            score += reward
            state = next_state  # Move to the next state
        
        return np.max(state), score
    
    def train(self, episodes, verbose_freq=5, performance_marks=[], num_trials=50):
        """Train the agent using temporal difference learning."""
        info = []
        idx = 0
        mark = -1 if not performance_marks else performance_marks[idx]
        
        for episode in range(episodes):
            tile, score = self.play_game()
            
            info.append((tile, score))
            if (episode + 1) % verbose_freq == 0:
                print(f'Episode {episode + 1}/{episodes}: Score {score}, Max {tile}')
                
            if mark == episode + 1:
                play_info = []
                self.set_learning(False)
                for i in range(num_trials):
                    tile, score = self.play_game(test=True)
                    play_info.append((tile, score))
                    print(f'Game {i+1}/{num_trials}: Score {score}, Max {tile}')
                
                play_info = np.array(play_info)
                
                freq = self.plot_info(play_info, title=f"Distribution at Episode {mark}")
                self.play_freqs.append(freq)
                
                idx += 1
                mark = performance_marks[min(idx, len(performance_marks) - 1)]
            
                self.set_learning(True)
        
    def make_video(self, video_name):
        self.set_learning(False)
        screen_size = 512
        pygame.init()
        screen = pygame.display.set_mode((screen_size, screen_size))
        clock = pygame.time.Clock()
        
        font = pygame.freetype.SysFont("font", 24)
        
        move_count = 0

        done = False
        score = 0
        
        state = self.env.reset()
        while not done:
            # Choose the best action (arg max of evaluate
            Visuals.draw_board(self.env.board)
            frame_path = os.path.join(newpath, f"Screenshots/frame_{move_count:05d}.png")
            pygame.image.save(screen, frame_path)

            action = self.policy(state)
            
            reward, done, next_state, next_states = self.make_move(state, action, display=False)
            
            if self.learning_enabled:
                self.td_update(state, action, reward, next_state)
            
            score += reward
            state = next_state  # Move to the next state
            move_count += 1
        #print(move_count)
                
        # Combine screenshots into a video using OpenCV
        video_filename = os.path.join(newpath, video_name)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(video_filename, fourcc, move_count // 25, (screen_size, screen_size))

        for i in range(move_count):
            frame_path = os.path.join(newpath, f"Screenshots/frame_{i:05d}.png")
            frame = cv2.imread(frame_path)
            out.write(frame)

        out.release()

        # Cleanup the screenshots directory
        for i in range(move_count):
            frame_path = os.path.join(newpath, f"Screenshots/frame_{i:05d}.png")
            os.remove(frame_path)

        print(f"Video saved as {video_filename}")
        
    
    def get_value(self, state):
        """Get the estimated value of the given state."""
        return self.get_state_value(state)
    
    def plot_info(self, info, title):
        freq = {}
        for i in info[:, 0]:
            freq[i] = freq.get(i, 0) + 1
        
        plt.bar([str(key) for key in freq.keys()], freq.values())
        plt.title(title)
        plt.show()
        return freq
        
import pickle
import os

def save_agent(agent, filename):
    """Save the trained agent to a file."""
    with open(filename, 'wb') as f:
        pickle.dump(agent, f)
    print(f"Agent saved to {filename}")

def load_agent(filename):
    """Load a trained agent from a file."""
    with open(filename, 'rb') as f:
        agent = pickle.load(f)
    print(f"Agent loaded from {filename}")
    agent.set_learning(False)
    print('Agent learning disabled for convience')
    return agent

            
# Initialize environment and agent
def train():
    env = Board.Gym2048Env(size=4, goal_power=2**14)
    
    #Change here based on whatsapp message
    agent = TDAfterStateAgent(env, epsilon=0.01)
    
    # Train agent over 1000 episodes
    agent.train(1000, verbose_freq=20, performance_marks=[100, 250, 500, 1000])

    save_agent(agent, os.path.join(newpath, 'Agent.pkl'))
    agent.make_video('training_video.mp4')
    
train()
# Test the trained agent

# import matplotlib.pyplot as plt
# info = np.array(info)
# plt.plot(np.arange(episodes), info[:, 1])
# plt.show()

# freq = {}
# for i in info[:, 0]:
#     freq[i] = freq.get(i, 0) + 1
    
# plt.bar([str(key) for key in freq.keys()], freq.values())
# plt.show()
# # Play a game using the trained Q-table
# agent.play()
