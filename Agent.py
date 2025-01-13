import os

# mypath = 'C:/Users/karab/Desktop/CS/Year 3/RL'
mypath = ''
import sys
if mypath not in sys.path and mypath != '':
    sys.path.append(mypath)
    
from Game import Game
import Visuals
import numpy as np

import pickle
import json
import math
# class Agent:
    
#     #4 rows + 4 cols + 9 2x2 squares = 17 in n_tuple with 12**4 state space
#     parameter_shape = (17, 12 ** 4)
    
#     def __init__(self):
#         self.feature_count, self.feature_size = Agent.parameter_shape
        
#         self.weights = (np.random.random((self.feature_count, self.feature_size)) / 100).tolist()
#         self.weight_signature = (self.feature_count,)
#         self
    
    # def n_tuple_4(self, state):
    #     x_ver = state
    #     x_hor = state.T
    #     x_sqr = np.lib.stride_tricks.sliding_window_view(state, (2, 2)).reshape(-1, 4)
        
    #     return np.concatenate((x_ver, x_hor, x_sqr))
    
#     def evaluate(self, row, score=None):
#         return sum([self.weights[i][f] for i, f in enumerate(self.n_tuple_4(state))])


class TDAgent:
    def __init__(self, game, alpha=0.25, min_alpha=0.01, alpha_decay = 0.994, epsilon=0.2, learning_enabled=True, episodes=None):  
        self.alpha = alpha # Learning rate
        self.min_alpha = min_alpha
        self.alpha_decay = alpha_decay
        self.epsilon = epsilon # exploration rate
         
        self.learning_enabled = learning_enabled
         
        # TODO: change this
        # self.game: Game = game
        self.game = game
         
        #4 rows + 4 cols + 9 2x2 squares = 17 in n_tuple with 12**4 state space
        self.n_tuple_len = 17
        self.target_power = 12
        self.weights = np.random.random((self.n_tuple_len, self.target_power ** 4)) / 100
        
        self.LUT = [{} for _ in range(self.n_tuple_len)]
        
        self.actions = list(range(4))
         
        self.info = []
        self.current_episodes = episodes if episodes else 0
        
         
    def n_tuple_4(self, state):
        x_ver = state
        x_hor = state.T
        x_sqr = np.lib.stride_tricks.sliding_window_view(state, (2, 2)).reshape(-1, 4)
        
        return np.concatenate((x_ver, x_hor, x_sqr))
     
    def compute_afterstate(self, state, direction):
        _state, reward, changed = self.game.next_state(state, direction)
        return _state, reward
    
    def update_V(self, state, delta):
        
        for _ in range(4):
            
            for i, f in enumerate(self.n_tuple_4(state)):
                self.weights[i][self.tuple_id(f)] += delta/8
            
            state = np.transpose(state)
            
            for i, f in enumerate(self.n_tuple_4(state)):
                self.weights[i][self.tuple_id(f)] += delta/8
            
            state = np.rot90(np.transpose(state))
            
        # for i in range(8):
        #     state = np.rot90(state, i % 4)
            
        #     if i == 4:
        #         state = state.T
            
        #     n_rep = self.n_tuple_4(state)
            
        #     for j, row in enumerate(n_rep):
        #         tpid = self.tuple_id(row)
        #         self.LUT[j][tpid] = self.LUT[j].get(tpid, 0) + delta
    
    def tuple_id(self, tp):
        n = 0
        k = 1
        
        for v in tp:
            n += v * k
            k *= self.target_power
        
        return n
            
        
    def V(self, state, delta=None):
        if delta is not None:
            
            self.update_V(state, delta)
        return np.mean([self.weights[i][self.tuple_id(f)] for i, f in enumerate(self.n_tuple_4(state))])
    
        # n_rep = self.n_tuple_4(state)
        
        # vals = []
        # for i, row in enumerate(n_rep):
        #     tpid = self.tuple_id(row)
            
        #     v = self.LUT[i].get(tpid, 0)
            
        #     if delta is not None:
        #         self.update_V(state, delta)
            
        #     vals.append(v)
        
        # return np.mean(vals)
            
    def evaluate(self, state, direction):
        _state, reward = self.compute_afterstate(state, direction)
        return reward + self.V(_state)
            

    def learn_evaluation(self, state, direction, reward, state1, state2, valid_moves):
        next_action = max(valid_moves, key=lambda a: self.evaluate(state2, a))
        
        next_state2, r_next = self.compute_afterstate(state2, next_action)
        v_next = self.V(next_state2)
        
        delta = r_next + v_next - self.V(state1)
        self.V(state1, delta = self.alpha * delta / self.n_tuple_len)
        
    def play_game(self):
        score = 0
        state = self.game.reset()
        
        while not self.game.is_finished(state):
            valid_moves = self.game.get_valid_moves(state)
            action = max(valid_moves, key=lambda a: self.evaluate(state, a))
            state1, state2, reward = self.game.make_move(action)
            
            if self.learning_enabled:
                self.learn_evaluation(state, action, reward, state1, state2, valid_moves)
            
            score += reward
            state = state2
        
        return 2 ** np.max(state), score
    
    def get_distribution(self, info):
        freq = {}
        for (max_tile, _) in info:
            freq[max_tile] = freq.get(max_tile, 0) + 1
        
        for key in freq:
            freq[key] = freq[key] / len(info) * 100
        
        avg_score = np.mean(np.array(info)[:, 1])
        
        return freq, avg_score
    
    def print_distribution(self, info):
        print(f'\n--------------------------\nDISTRIBUTION of {len(info)} trials:\n--------------------------')
        freq, avg_score = self.get_distribution(info)
        
        for key in sorted(freq.keys()):
            print(f'{key}: {freq[key]:.2f}%')
        print('--------------------------')
        print(f'{avg_score = }')
        print('--------------------------\n')
      
    def decay_alpha(self):
        self.alpha = max(self.min_alpha, self.alpha * self.alpha_decay)
        
    def train(self, episodes=10000, trial_freq=100, num_trial=200, verbose_freq=10):
        
        self.learning_enabled = True
        first_2048 = 0 
        training_winrates = []
        trial_info = []
        for episode in range(1, episodes+1):
            max_tile, score = self.play_game()
            if episode % verbose_freq == 0:
                #ddebugging purposes
                max_weight = np.max(self.weights)
                print(f'Episode {episode}/{episodes}: {max_tile = }, {score = }, {max_weight = }')
            
            if max_tile == 2048:
                first_2048 = episode
                print(f'First 2048 occured after {first_2048} episodes')
            
            self.info.append((max_tile, score))
            
            if episode % trial_freq == 0:
                current_info = np.array(self.info)
                winrate = len(np.where(current_info[:, 0] == 2048)) / num_trial
                training_winrates.append(winrate)
                
                self.learning_enabled = False
                current_trials = []
                print(f'Current Training: Episode {episode}.\nPlaying {num_trial} games...')
                for _ in range(num_trial):
                    max_tile, score = self.play_game()
                    current_trials.append((max_tile, score))
                
                self.learning_enabled = True
                trial_info.append(current_trials)
                self.print_distribution(current_trials)
            
            self.decay_alpha()
                
        
        self.print_distribution(self.info)
        return training_winrates, trial_info
    
    def play_game_with_states(self):
        score = 0
        state = self.game.reset()
        states = [state]
        while not self.game.is_finished(state):
            action = max(self.game.get_valid_moves(state), key=lambda a: self.evaluate(state, a))
            state1, state2, reward = self.game.make_move(action)
            
            score += reward
            state = state2
            states.append(state)
        
        return states, score
    
    def expectimax(self, state, depth, is_maximizing):
        if depth == 0 or self.game.is_finished(state):
            return np.sum(state)
        
        if is_maximizing:
            max_value = - math.inf
            
            for action in self.game.get_valid_moves(state):
                state1, state2, reward = self.game.next_state(state, action)
                max_value = max(max_value, self.expectimax(state1, depth-1, False))
                
            return max_value
            
        else:
            expected_val = 0
            for empty in self.game.empties(state):
                for p, val in [(0.1, 4), (0.9, 2)]:
                    state2 = np.copy(state)
                    state2[empty] = val
                    expected_val += p * self.expectimax(state2, depth-1, True)
                    
                        
            return expected_val
                    
    def play_with_expectimax(self, depth=5):
        score = 0
        state = self.game.reset()
        
        states = [state]
        while not self.game.is_finished(state):
            
            best_move = None
            best_val = -math.inf
            
            valid_moves = self.game.get_valid_moves(state)
            
            for action in valid_moves:
                state1, _, _ = self.game.next_state(state, action)
                move_val = self.expectimax(state1, depth-1, False)
                if move_val > best_val:
                    best_val = move_val
                    best_move = action
            
            _, state2, r = self.game.make_move(best_move)
            score += r
            
            state = state2
            states.append(state)
        
        return states, score
        
    def save_agent(self, filename):
        """Save the trained agent to a file."""
        with open(filename, 'wb') as f:
            pickle.dump(self, f)
            print(f"Agent saved to {filename}")
            
        
def load_agent(filename):
    """Load a trained agent from a file."""
    with open(filename, 'rb') as f:
        agent = pickle.load(f)
    print(f"Agent loaded from {filename}")
    agent.set_learning(False)
    print('Agent learning disabled for convience')
    return agent        

agent_path = os.path.join(mypath, 'agent.pkl')
game = Game()

#change these two lines
#agent = load_agent(agent_path)
agent = TDAgent(game)

no_training_game, score = agent.play_game_with_states()

output_path = 'no_training.mp4'
output_path = os.path.join(mypath, output_path)
visuals = Visuals.Visuals(no_training_game, output_video=output_path)
visuals.generate_gameplay_video()
visuals.pyquit()



winrates, trialss = agent.train(episodes=100_000, trial_freq=1000, verbose_freq=50)

winrate_path = 'winrate.txt'
np.savetxt(os.path.join(mypath, winrate_path), np.array(winrates))

trials_path = os.path.join(mypath, 'trials')
try:
    os.mkdir(trials_path)
    print(f"Directory '{trials_path}' created successfully.")
except FileExistsError:
    print(f"Directory '{trials_path}' already exists.")
for i, trials in enumerate(trialss):
    trial_path = f'trials{i}.txt'
    np.savetxt(os.path.join(trials_path, trial_path), np.array(trials))

agent.save_agent(os.path.join(mypath, 'agent.pkl'))

max_training_game, score = agent.play_game_with_states()
output_path = 'max_training.mp4'
output_path = os.path.join(mypath, output_path)
visuals = Visuals.Visuals(max_training_game, output_video=output_path)
visuals.generate_gameplay_video()
visuals.pyquit()
