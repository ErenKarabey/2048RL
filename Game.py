import numpy as np
import pickle
import random

def create_table():
    """
    Creates a lookup table for every possible row
    to left_shifted (row', reward).
    Ex:
        row: 4 4-bit tuple - 0 1 1 2
        row': 4 4-bit integer tuple - 2 2 0 0
        reward: integer - 2 + 2 = 4
    """
    table = {}
    for b1 in range(12):
        for b2 in range(12):
            for b3 in range(12):
                for b4 in range(12):
                    row = (b1, b2, b3, b4)
                    table[row] = _left(row)
    
    return table

def _left(row):
    """
    Perform a left shift operation on a 1D array of 4 integers for a 2048 game.

    Args:
        row (list): A list of 4 integers representing a row in the game.

    Returns:
        row' (list): The updated row after performing the left shift.
        reward (int): sum of merges
        changed (bool): whether the moved changed or not
    """
    # Remove zeroes and keep non-zero elements
    non_zero = [num for num in row if num != 0]
    
    # Combine adjacent equal elements
    reward = 0
    new_row = []
    skip = False
    for i in range(len(non_zero)):
        if skip:
            skip = False
            continue
        if i < len(non_zero) - 1 and non_zero[i] == non_zero[i + 1]:
            new_row.append(non_zero[i] + 1)
            reward += 2 ** (non_zero[i] + 1) 
            skip = True
        else:
            new_row.append(non_zero[i])
    
    # Pad with zeroes to maintain length of 4
    while len(new_row) < 4:
        new_row.append(0)
    
    return tuple(new_row), reward, tuple(new_row) != tuple(row)

            

class Game:
    
    actions = {0: 'left', 1: 'up', 2: 'right', 3: 'down'}
    table = create_table()
    counter = 0
    save_file = 'saved_game.pkl'
    
    def __init__(self, score=0, state=None):
        
        self.score = score
        self.moves = []
        
        if state is None:
            self.state = np.zeros((4,4), np.int8)
            
            self.spawn_tile()
            self.spawn_tile()
            
    def save_game(self, file=None):
        file = file or self.file
        with open(file, 'wb') as f:
            pickle.dump(self, f, -1)
    
    def load_game(file=save_file):
        with open(file, 'rb') as f:
            game = pickle.load(f)
        return game
    
    def empties(self, state):
        zeros = np.where(state == 0)
        return list(zip(zeros[0], zeros[1]))
    
    def adj_pair_count(self, state):
        return 24 - np.count_nonzero(state[:, :3] - state[:, 1:]) - np.count_nonzero(state[:3, :] - state[1:, :])
    
    def is_finished(self, state):
        return (not self.empties(state) and self.adj_pair_count(state) == 0) or np.max(state) == 12
    
    def create_tile(self, state):
        emptis = self.empties(state)
        
        tile = 1
        if np.random.random() > 0.9:
            tile = 2
        
        return (tile, random.choice(emptis))
    
    def spawn_tile(self):
        tile, pos = self.create_tile(self.state)
        self.state[pos] = tile        
        
    def move_left(self, state):
        total_reward = 0
        _state = []
        changed = False
        for row in state:
            _row, reward, _changed = Game.table[tuple(row)]
            total_reward += reward
            changed = changed or _changed
            _state.append(_row)
        
        if total_reward == 0:
            total_reward = -5 # Negative reward for illegal move
        
            
        return _state, total_reward, changed
        
    def next_state(self, state, direction):
        _state = np.rot90(state, direction)
        _state, reward, changed = self.move_left(_state)
        _state = np.rot90(_state, 4 - direction)
        
        return _state, reward, changed
    
    def get_valid_moves(self, state):
        actions = []
        for i in range(4):
            _, _, changed = self.next_state(state, i)
            if changed:
                actions.append(i)
        return actions
    
    def make_move(self, direction):
        Game.counter += 1
        self.moves.append(direction)
        
        self.state, reward, changed = self.next_state(self.state, direction)
        state = self.state
        if changed:
            self.spawn_tile()
            self.moves.append(direction)
        else:
            print("Invalid Move!")
        self.score += reward
        return state, self.state, reward
    
    def reset(self):
        self.state = np.zeros((4,4), dtype=np.int8)
        
        self.spawn_tile()
        self.spawn_tile()
        
        self.score = 0
        self.moves = []
        
        return self.state
    
    
    def __str__(self):
        return str(np.matrix(self.state))
    
    def __repr__(self):
        return str(np.matrix(self.state))        
