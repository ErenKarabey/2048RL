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
