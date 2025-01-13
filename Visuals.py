import pygame
import cv2
import numpy as np
import random


WIDTH, HEIGHT = 400, 400
GRID_SIZE = 4
TILE_SIZE = WIDTH // GRID_SIZE
BACKGROUND_COLOR = (187, 173, 160)  # light brown background for grid
TILE_COLORS = {
    2: (238, 228, 218),
    4: (237, 224, 200),
    8: (242, 177, 121),
    16: (245, 149, 99),
    32: (246, 124, 95),
    64: (246, 94, 59),
    128: (237, 207, 114),
    256: (237, 204, 97),
    512: (237, 200, 80),
    1024: (237, 197, 63),
    2048: (237, 194, 46),
}
# Key note is that 0 is same as background which looks really cool
TEXT_COLOR = (119, 110, 101)

class Visuals:
    def __init__(self, board_states, output_video="2048_gameplay.mp4", frame_rate=10):
        """ Initialize the Visuals class with game board states and video settings. """
        pygame.init()

        # Game settings
        self.board_states = board_states
        self.output_video = output_video
        self.frame_rate = frame_rate
        self.WIDTH = WIDTH
        self.HEIGHT = HEIGHT
        self.GRID_SIZE = GRID_SIZE
        self.TILE_SIZE = TILE_SIZE
        
        # Initialize screen and font
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption('2048 Gameplay')
        self.font = pygame.font.SysFont("Arial", 40)

        # Setup for video output
        self.fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.out = cv2.VideoWriter(output_video, self.fourcc, frame_rate, (WIDTH, HEIGHT))
        

    def draw_grid(self):
        """ Draw the grid of the game. """
        for i in range(self.GRID_SIZE):
            for j in range(self.GRID_SIZE):
                pygame.draw.rect(self.screen, BACKGROUND_COLOR, (i * self.TILE_SIZE, j * self.TILE_SIZE, self.TILE_SIZE, self.TILE_SIZE))
                pygame.draw.rect(self.screen, (187, 173, 160), (i * self.TILE_SIZE, j * self.TILE_SIZE, self.TILE_SIZE, self.TILE_SIZE), 5)

    def draw_tile(self, value, row, col):
        """ Draw a tile on the grid. """
        if value != 0:
            tile_color = TILE_COLORS.get(value, (60, 58, 50))
            pygame.draw.rect(self.screen, tile_color, (col * self.TILE_SIZE, row * self.TILE_SIZE, self.TILE_SIZE, self.TILE_SIZE))
            text = self.font.render(str(value), True, TEXT_COLOR)
            text_rect = text.get_rect(center=(col * self.TILE_SIZE + self.TILE_SIZE // 2, row * self.TILE_SIZE + self.TILE_SIZE // 2))
            self.screen.blit(text, text_rect)

    def draw_game(self, grid):
        """ Draw the full game state. """
        self.screen.fill((250, 248, 239))  # Set background color
        self.draw_grid()
        for i in range(self.GRID_SIZE):
            for j in range(self.GRID_SIZE):
                val = 2 ** grid[i][j] if grid[i][j] else 0
                self.draw_tile(val, i, j)

        # Capture the current frame for video
        frame = pygame.surfarray.array3d(pygame.display.get_surface())  # Get screen content as an array
        frame = np.transpose(frame, (1, 0, 2))  # Transpose to match OpenCV format (height, width, channels)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # Convert color format from RGB to BGR
        self.out.write(frame)  # Write the frame to the video

    def generate_gameplay_video(self):
        """ Generate the gameplay video from the list of board states. """
        for grid in self.board_states:
            self.draw_game(grid)
        
        # Release the video writer
        self.out.release()

    def pyquit(self):
        """ Quit Pygame. """
        pygame.quit()


