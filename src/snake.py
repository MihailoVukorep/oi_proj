from .globals import *

class Snake:
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.body = [(GRID_WIDTH // 2, GRID_HEIGHT // 2)]
        self.direction = RIGHT
        self.grow = False
        self.alive = True
        self.score = 0
        self.steps = 0
        self.steps_since_food = 0
        
    def move(self):
        if not self.alive:
            return
            
        head_x, head_y = self.body[0]
        dx, dy = self.direction
        new_head = (head_x + dx, head_y + dy)
        
        # Check wall collision
        if (new_head[0] < 0 or new_head[0] >= GRID_WIDTH or 
            new_head[1] < 0 or new_head[1] >= GRID_HEIGHT):
            self.alive = False
            return
        
        # Check self collision
        if new_head in self.body:
            self.alive = False
            return
        
        self.body.insert(0, new_head)
        
        if not self.grow:
            self.body.pop()
        else:
            self.grow = False
            self.score += 1
            self.steps_since_food = 0
        
        self.steps += 1
        self.steps_since_food += 1
        
        # Die if taking too long without food
        if self.steps_since_food > 400:
            self.alive = False
    
    def set_direction(self, direction):
        # Prevent moving backwards
        if (direction[0] * -1, direction[1] * -1) != self.direction:
            self.direction = direction

    def check_self_collision(self):
        """Check if the snake has collided with itself"""
        head = self.body[0]
        return head in self.body[1:]
    
    def eat_food(self):
        self.grow = True
