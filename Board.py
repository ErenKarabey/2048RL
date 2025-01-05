# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 12:50:24 2024

@author: karab
"""

import gymnasium as gym
import numpy as np

# TODO: Fix the shifting algorithm

# board = np.random.randint(0, 5, (4,4))

# board = 2 ** board
# board[board == 1] = 0

# def _right(arr):
    
#     new_arr = []
#     zero_count = 0
    
#     for i in range(len(arr)):
        
#         if arr[i] == 0:
#             zero_count += 1
#             continue
        
#         if new_arr:
#             last = new_arr[-1]
#             if last == arr[i]:
#                 new_arr[-1] *= 2
#                 zero_count += 1
#             else:
#                 new_arr.append(arr[i])
#         else:
#             new_arr.append(arr[i])
    
#     return [0 for _ in range(zero_count)] + new_arr

class Gym2048Env(gym.Env):
    metadata = {"rendermodes": ["human"]}
    
    def __init__(self, size=4, goal_power=2048, render_mode=None):
        self.size = size
        self.goal = goal_power
        self.failed = False
        
        assert size > 1, "The dimensions of the grid must be at least 2!"
        
        # Not reward but actual game score
        self.current_score = 0
        
        
        
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
        
        self.seed(seed=1)
        
        self.reset()
    
    def seed(self, seed=None):
        """Sets random seed for gym.Env."""
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]
    
    def step(self, direction):
        """
        Move based on direction, spawn new tile and update score
        """
        
        self.move(direction)
        
        self.score = np.max(self.board)
        
        self.spawn_tile()
        self.print_board()
    
    def move(self, direction):
        """
        Move based on direction
        0: Slide-Up, 1: Slide-Left, 2: Slide-Down, 3:Slide-Right
        """
        self._dir_to_function[direction]()
        
    def _left(self):
        """Apply the move 'left' """
        self.board = np.apply_along_axis(self.shift1D, 1, self.board, 'l')
        #np.flip caused the "2 2 4 8" -> "0 0 0 16" error so i rewrote the function to do a left or right shift - jay
    
    def _right(self):
        """Apply the move 'right' """
        self.board = np.apply_along_axis(self.shift1D, 1, self.board, 'r')
        
    def _up(self):
        """Apply the move 'up' """
        self.board = self.board.T
        self._left()
        self.board = self.board.T
    
    def _down(self):
        """Apply the move 'down' """
        self.board = self.board.T
        self._right()
        self.board = self.board.T
    
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
        return self.failed
    
    def spawn_tile(self):
        """Add tiles after each move. 80% chance of 2 and 20% chance of 4"""
        if self.np_random.random() < 0.8:
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
        
        self.current_score = 0
        
        self.spawn_tile()
        self.spawn_tile()
        
    
    def print_board(self):
        """Print board with no borders"""
        txt = str(np.matrix(self.board, dtype = np.int32))
        print(txt.replace(']', '').replace('[', ' '))
        

        
