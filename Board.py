# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 12:50:24 2024

@author: karab
"""

import gymnasium as gym
import numpy as np


class Gym2048Env(gym.Env):
    metadata = {"rendermodes": ["human"]}
    
    def __init__(self, size=4, goal_power=2048, render_mode=None, seed=None):
        self.size = size
        self.goal = goal_power
        self.failed = False
        
        assert size > 1, "The dimensions of the grid must be at least 2!"
        
        # Not reward but actual game score
        self.score = 0
        
        
        
        # These are the inherithances from gym.Env
        # 0: Slide-Up, 1: Slide-Left, 2: Slide-Down, 3:Slide-Right
        self.action_space = gym.spaces.Discrete(4)
        
        self._dir_to_function = [self._up, self._left, self._down, self._right]
        
        self.observation_space = gym.spaces.Box(
            low=0,
            high=2**(self.size * self.size), #Max power could be assume to 2^square_count
            shape=(size, size),
            dtype=np.int32,
        )
        
        self.board = np.zeros((size, size))
        
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        
        self.seed(seed=seed)
        
        self.reset()
    
    def seed(self, seed=None):
        """Sets random seed for gym.Env."""
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def get_next_states(self):
        next_states = []
        
        self.reward = 0
        
        for action in range(4):
            next_board = self.move(self.board, action)
            reward = self.reward
            
            if np.all(self.board == next_board):
                continue
            
            next_states.append((action, reward, next_board))
        
        return next_states
    
    def step(self, direction, display=True, spawn=True):
        """
        Move based on direction, spawn new tile and update score
        Assume nonillegal move
        """
        
        self.reward = 0
        self.board = self.move(self.board, direction)
        
        self.score += self.reward
        
        if display: self.print_board()
        
        if spawn: self.spawn_tile()
        
        return (self.board, self.reward, self.is_finished(), self.score)
    
    def move(self, board, direction):
        """
        Move based on direction
        0: Slide-Up, 1: Slide-Left, 2: Slide-Down, 3:Slide-Right
        """
        return self._dir_to_function[direction](board)
        
    def _left(self, board):
        """Apply the move 'left' """
        self.reward = 0
        return np.apply_along_axis(self.shift1D, 1, board, 'l')
        #np.flip caused the "2 2 4 8" -> "0 0 0 16" error so i rewrote the function to do a left or right shift - jay
    
    def _right(self, board):
        """Apply the move 'right' """
        self.reward = 0
        return np.apply_along_axis(self.shift1D, 1, board, 'r')
        
    def _up(self, board):
        """Apply the move 'up' """
        self.reward = 0
        board = self._left(board.T)
        return board.T
    
    def _down(self, board):
        """Apply the move 'down' """
        self.reward = 0
        board = self._right(board.T)
        return board.T
    
    def shift1D(self, arr, direction): #made this a more general function since slightly different things have to happen depending on left / right. - jay
        """1D shift helper function"""
        new_arr = []
        row_len = len(arr)

        #Get non-empty tiles
        for i in range(row_len):
            if arr[i] != 0:
                new_arr.append(arr[i])

        #combine tiles pairwise
        iterate_order = range(len(new_arr) - 1)
        if direction == 'r': iterate_order = iterate_order[::-1]
        
        for i in iterate_order:
            #if two same tiles next to each other, combine
            if new_arr[i] == new_arr[i + 1]:
                new_arr[i] = new_arr[i] * 2                
                new_arr[i + 1] = 0
                
                #Update score by newly merged value
                self.reward += new_arr[i]

        #remove leftover empty (0) tiles
        new_arr = [v for v in new_arr if v != 0]

        if direction == 'r':   #if shifting right,
            return [0 for _ in range(row_len - len(new_arr))] + new_arr
        
        elif direction == 'l': #if shifting left,
            return new_arr + [0 for _ in range(row_len - len(new_arr))]
        
        else:                  #if unknown direction (not l or r)
            print("Unknown direction: should be 'l' or 'r'. Defaulted to 'r'")
            return [0 for _ in range(row_len - len(new_arr))] + new_arr

    
    def is_finished(self):
        if np.max(self.board) == self.goal or self.failed:
            return True
        
        for i in range(self.size):
            for j in range(self.size):
                if self.board[i][j] == 0:
                    return False
                if i != 0 and self.board[i - 1][j] == self.board[i][j]:
                    return False
                if j != 0 and self.board[i][j - 1] == self.board[i][j]:
                    return False
        return True
    
    def spawn_tile(self):
        # This should be 90% i think
        """Add tiles after each move. 90% chance of 2 and 10% chance of 4"""
        if self.np_random.random() < 0.9:
            tile = 2
        else: tile = 4

        #If nowhere to place tile, then game is over!
        if not (0 in self.board):
            self.failed = True
            return
        
        empties_x, empties_y = np.where(self.board == 0)

        idx = self.np_random.choice(len(empties_x))
        self.board[empties_x[idx], empties_y[idx]] = tile
        
        
    def reset(self, seed=None):
        """Reset the board and spawn 2 new tiles - 2 is hardcoded"""
        self.board = np.zeros((self.size, self.size), dtype=np.int32)
        
        self.score = 0
        
        self.spawn_tile()
        self.spawn_tile()
        
        self.failed = False
        
        return self.board
        
    
    def print_board(self):
        """Print board with no borders"""
        txt = str(np.matrix(self.board, dtype = np.int32))
        print(txt.replace(']', '').replace('[', ' '))
        print(f'Score: {self.score}')
        

        
# For test
test = Gym2048Env()
