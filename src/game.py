import numpy as np
import random
import globals as gl
import snake

class Game:
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.snake = snake.Snake()
        self.place_food()
        self.game_over = False
    
    def place_food(self):
        while True:
            x = random.randint(0, gl.GRID_WIDTH - 1)
            y = random.randint(0, gl.GRID_HEIGHT - 1)
            if (x, y) not in self.snake.body:
                self.food = (x, y)
                break
    
    def step(self, action):
        if self.game_over:
            return
        
        # Convert action to direction
        if action == 0:  # Up
            self.snake.set_direction(gl.UP)
        elif action == 1:  # Down
            self.snake.set_direction(gl.DOWN)
        elif action == 2:  # Left
            self.snake.set_direction(gl.LEFT)
        elif action == 3:  # Right
            self.snake.set_direction(gl.RIGHT)
        
        self.snake.move()
        
        # Check food collision
        if self.snake.alive and self.snake.body[0] == self.food:
            self.snake.eat_food()
            self.place_food()
        
        if not self.snake.alive:
            self.game_over = True
    
    def get_state(self):
        """Get current game state for neural network input"""
        head_x, head_y = self.snake.body[0]
        food_x, food_y = self.food
        
        # Distance to food
        food_dist_x = food_x - head_x
        food_dist_y = food_y - head_y
        
        # Danger detection (straight, left, right relative to current direction)
        danger_straight = self.is_collision(head_x, head_y, self.snake.direction)
        
        # Get left and right directions relative to current direction
        dir_idx = gl.DIRECTIONS.index(self.snake.direction)
        left_dir = gl.DIRECTIONS[(dir_idx - 1) % 4]
        right_dir = gl.DIRECTIONS[(dir_idx + 1) % 4]
        
        danger_left = self.is_collision(head_x, head_y, left_dir)
        danger_right = self.is_collision(head_x, head_y, right_dir)
        
        # Current direction as one-hot
        dir_up = self.snake.direction == gl.UP
        dir_down = self.snake.direction == gl.DOWN
        dir_left = self.snake.direction == gl.LEFT
        dir_right = self.snake.direction == gl.RIGHT
        
        # Food direction relative to snake
        food_up = food_y < head_y
        food_down = food_y > head_y
        food_left = food_x < head_x
        food_right = food_x > head_x
        
        state = [
            danger_straight,
            danger_left,
            danger_right,
            dir_up,
            dir_down,
            dir_left,
            dir_right,
            food_up,
            food_down,
            food_left,
            food_right
        ]
        
        return np.array(state, dtype=np.float32)
    
    def is_collision(self, x, y, direction):
        """Check if moving in direction from position would cause collision"""
        dx, dy = direction
        new_x, new_y = x + dx, y + dy
        
        # Wall collision
        if new_x < 0 or new_x >= gl.GRID_WIDTH or new_y < 0 or new_y >= gl.GRID_HEIGHT:
            return True
        
        # Self collision
        if (new_x, new_y) in self.snake.body:
            return True
        
        return False
