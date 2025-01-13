# 2048RL
RL agent for 2048

To run your own agent simply run the Agent.py with your chosen hyperparameters
This will firstly create a `no_training.mp4` video of agent with no training playing one game. This is highly random as `self.weights` are initilized randomly.
Then it will train for your given `episodes`, and create `max_training.mp4` video.
Lastly based on `agent.train()` parameters such as `trial_freq` and `num_trial` it will run `num_trial` games every `trial_freq` episodes and record the winrate and trials data as `winrate.txt` and `trials/trial<num>.txt` respectively. This formatting is neccesary for trials as the result is a 3D array and `np.savetxt` does not work properly in array-dimension greater than 2. 
