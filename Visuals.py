#Draws board using pygame
#By Jay Hilton

import pygame
import pygame.freetype

#Set up pygame
screen_size = 512
pygame.init()
screen = pygame.display.set_mode((screen_size, screen_size))
clock = pygame.time.Clock()

font = pygame.freetype.SysFont("font", 24)

#warining!
print("If you want to close the display, use close_display() or close this shell window.")

#vars
active = True

# - colours
col_background = "burlywood4"

col_0 = "burlywood1"
col_2 = "cornsilk4"
col_4 = "darkorange4"
col_8 = "darkorange2"
col_16 = "orange2"
col_32 = "orange"
col_64 = "red2"
col_128 = "red"
col_256 = "violetred3"
col_512 = "violetred1"
col_1024 = "orchid4"
col_2048 = "orchid"

col_undefined = "blue"

# -=- modules -=-

#draws the current board to the screen
def draw_board(board):
    if not active:
        return

    #stop from freezing
    pygame.event.pump()

    #Draw board
    draw_background()

    #calculate tile and border size
    size = len(board)
    border_size, tile_size = calc_tile_border(size)

    #Draw tiles
    x_pos = 0
    y_pos = 0
    for tile_x in range(size):
        x_pos += border_size
        for tile_y in range(size):
            y_pos += border_size

            #Draw tile
            pos = (x_pos, y_pos)
            value = board[tile_y][tile_x]
            draw_tile(pos, tile_size, value)

            y_pos += tile_size
        x_pos += tile_size
        y_pos = 0
            

    #put changes onto screen 
    update_display()
    
    return

#draws solid colour background
def draw_background():
    screen.fill(col_background)
    return

#calculates tile and border size
def calc_tile_border(board_size):
    border_size = screen_size / ( (8 * board_size) + 1)
    tile_size = border_size * 7

    return (border_size, tile_size)

#Draw one tile
def draw_tile(pos, size, value):

    #Rect
    rect = pygame.Rect(pos, (size, size))
    col = value_to_colour(value)
    pygame.draw.rect(screen, col, rect)

    #Text
    x, y = pos
    text_pos = (x + (size / 2) - (6 * len(str(value))), y + (size / 2) - 12)
    text_surface, rect = font.render(str(value), (0, 0, 0))
    screen.blit(text_surface, text_pos)
    

#Turn tile value into colour
def value_to_colour(value):
    if value == 0: return col_0
    if value == 2: return col_2
    if value == 4: return col_4
    if value == 8: return col_8
    if value == 16: return col_16
    if value == 32: return col_32
    if value == 64: return col_64
    if value == 128: return col_128
    if value == 256: return col_256
    if value == 512: return col_512
    if value == 1024: return col_1024
    if value == 2048: return col_2048
    return col_undefined

def update_display():
    pygame.display.flip()

def close_display():
    active = False
    pygame.quit()

