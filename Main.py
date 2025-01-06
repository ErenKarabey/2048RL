import gymnasium as gym
import Board
import Visuals

show_visuals = True

env = Board.Gym2048Env()

if not show_visuals:
    Visuals.close_display()
    


#play game manually. enjoy! - jay
env.print_board()
while True:
    if show_visuals:
        Visuals.draw_board(env.board)
    
    inp = input(">")
    if   inp == "u": action = 0
    elif inp == "l": action = 1
    elif inp == "d": action = 2
    elif inp == "r": action = 3
    else: 
        try:
            action = int(inp)
            assert 0 <= action <= 3, "Invalid input!"
        except:
            Board.close_display()
            print("Invalid input!")
            break
    
    env.step(action)

def random_play(game_count):
    game_count = 1000
    scores = []

    i = 0
    while game_count > i:
        
        if show_visuals:
            Visuals.draw_board(env.board)
            
        while not env.is_finished():
            action = np.random.randint(0, 4)
            env.step(action, display=False)
                
            scores.append((np.max(env.board), env.score))
            env.reset()
            i += 1
                
                
    scores = np.array(scores)
    game_scores = scores[:, 1]
                
    plt.bar(np.arange(game_count), game_scores)
    plt.title("Game Scores")
    plt.ylabel('Game Score')
    plt.show()
                
    powers = {2**i:0 for i in range(1, 12)}
    for _max in scores[:, 0]:
        powers[_max] += 1
                    
        plt.bar([str(key) for key in powers.keys()], powers.values())
        plt.title('Max Tile Frequency')
        plt.xlabel('Max Tile')
        plt.ylabel('Frequency')
        plt.show()

