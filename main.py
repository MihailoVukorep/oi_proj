import numpy as np
import random
import pygame
import sys
from typing import List, Tuple, Optional
import math

# Game constants
GRID_SIZE = 20
GRID_WIDTH = 20
GRID_HEIGHT = 20
CELL_SIZE = 20
WINDOW_WIDTH = GRID_WIDTH * CELL_SIZE
WINDOW_HEIGHT = GRID_HEIGHT * CELL_SIZE

# Colors
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
WHITE = (255, 255, 255)

# Directions
UP = (0, -1)
DOWN = (0, 1)
LEFT = (-1, 0)
RIGHT = (1, 0)
DIRECTIONS = [UP, DOWN, LEFT, RIGHT]

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
        if self.steps_since_food > 100:
            self.alive = False
    
    def set_direction(self, direction):
        # Prevent moving backwards
        if (direction[0] * -1, direction[1] * -1) != self.direction:
            self.direction = direction
    
    def eat_food(self):
        self.grow = True

class Game:
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.snake = Snake()
        self.place_food()
        self.game_over = False
    
    def place_food(self):
        while True:
            x = random.randint(0, GRID_WIDTH - 1)
            y = random.randint(0, GRID_HEIGHT - 1)
            if (x, y) not in self.snake.body:
                self.food = (x, y)
                break
    
    def step(self, action):
        if self.game_over:
            return
        
        # Convert action to direction
        if action == 0:  # Up
            self.snake.set_direction(UP)
        elif action == 1:  # Down
            self.snake.set_direction(DOWN)
        elif action == 2:  # Left
            self.snake.set_direction(LEFT)
        elif action == 3:  # Right
            self.snake.set_direction(RIGHT)
        
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
        dir_idx = DIRECTIONS.index(self.snake.direction)
        left_dir = DIRECTIONS[(dir_idx - 1) % 4]
        right_dir = DIRECTIONS[(dir_idx + 1) % 4]
        
        danger_left = self.is_collision(head_x, head_y, left_dir)
        danger_right = self.is_collision(head_x, head_y, right_dir)
        
        # Current direction as one-hot
        dir_up = self.snake.direction == UP
        dir_down = self.snake.direction == DOWN
        dir_left = self.snake.direction == LEFT
        dir_right = self.snake.direction == RIGHT
        
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
        if new_x < 0 or new_x >= GRID_WIDTH or new_y < 0 or new_y >= GRID_HEIGHT:
            return True
        
        # Self collision
        if (new_x, new_y) in self.snake.body:
            return True
        
        return False

class NeuralNetwork:
    def __init__(self, input_size=11, hidden_size=16, output_size=4):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Initialize weights randomly
        self.weights1 = np.random.randn(input_size, hidden_size) * 0.5
        self.bias1 = np.random.randn(hidden_size) * 0.5
        self.weights2 = np.random.randn(hidden_size, output_size) * 0.5
        self.bias2 = np.random.randn(output_size) * 0.5
    
    def forward(self, x):
        """Forward pass through network"""
        # First layer
        z1 = np.dot(x, self.weights1) + self.bias1
        a1 = np.tanh(z1)  # Activation function
        
        # Second layer
        z2 = np.dot(a1, self.weights2) + self.bias2
        a2 = z2  # Linear output
        
        return a2
    
    def predict(self, x):
        """Get action from network output"""
        output = self.forward(x)
        return np.argmax(output)
    
    def get_weights(self):
        """Get all weights as a flat array"""
        return np.concatenate([
            self.weights1.flatten(),
            self.bias1.flatten(),
            self.weights2.flatten(),
            self.bias2.flatten()
        ])
    
    def set_weights(self, weights):
        """Set weights from flat array"""
        idx = 0
        
        # First layer weights
        w1_size = self.input_size * self.hidden_size
        self.weights1 = weights[idx:idx + w1_size].reshape(self.input_size, self.hidden_size)
        idx += w1_size
        
        # First layer bias
        self.bias1 = weights[idx:idx + self.hidden_size]
        idx += self.hidden_size
        
        # Second layer weights
        w2_size = self.hidden_size * self.output_size
        self.weights2 = weights[idx:idx + w2_size].reshape(self.hidden_size, self.output_size)
        idx += w2_size
        
        # Second layer bias
        self.bias2 = weights[idx:idx + self.output_size]
    
    def copy(self):
        """Create a copy of this network"""
        new_net = NeuralNetwork(self.input_size, self.hidden_size, self.output_size)
        new_net.set_weights(self.get_weights())
        return new_net

class GeneticAlgorithm:
    def __init__(self, population_size=50, mutation_rate=0.1, mutation_strength=0.1):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.mutation_strength = mutation_strength
        self.generation = 0
        
        # Create initial population
        self.population = []
        for _ in range(population_size):
            network = NeuralNetwork()
            self.population.append(network)
        
        self.fitness_scores = []
        self.best_fitness = 0
        self.best_network = None
    
    def evaluate_fitness(self, network, games=3):
        """Evaluate fitness of a network by playing multiple games"""
        total_score = 0
        total_steps = 0
        
        for _ in range(games):
            game = Game()
            steps = 0
            
            while not game.game_over and steps < 1000:  # Max steps per game
                state = game.get_state()
                action = network.predict(state)
                game.step(action)
                steps += 1
            
            # Fitness calculation
            score = game.snake.score
            fitness = score * 100  # Base score
            
            # Bonus for staying alive longer
            fitness += steps * 0.1
            
            # Penalty for dying without eating
            if score == 0:
                fitness -= 50
            
            total_score += fitness
            total_steps += steps
        
        return total_score / games
    
    def select_parents(self, fitness_scores):
        """Select parents using tournament selection"""
        tournament_size = 5
        parents = []
        
        for _ in range(2):
            tournament = random.sample(range(len(fitness_scores)), tournament_size)
            winner = max(tournament, key=lambda x: fitness_scores[x])
            parents.append(self.population[winner])
        
        return parents
    
    def crossover(self, parent1, parent2):
        """Create offspring using crossover"""
        child = NeuralNetwork()
        
        weights1 = parent1.get_weights()
        weights2 = parent2.get_weights()
        
        # Single point crossover
        crossover_point = random.randint(1, len(weights1) - 1)
        child_weights = np.concatenate([
            weights1[:crossover_point],
            weights2[crossover_point:]
        ])
        
        child.set_weights(child_weights)
        return child
    
    def mutate(self, network):
        """Mutate network weights"""
        weights = network.get_weights()
        
        for i in range(len(weights)):
            if random.random() < self.mutation_rate:
                weights[i] += random.gauss(0, self.mutation_strength)
        
        network.set_weights(weights)
        return network
    
    def evolve(self):
        """Evolve the population for one generation"""
        print(f"Generation {self.generation}: Evaluating fitness...")
        
        # Evaluate fitness for all networks
        self.fitness_scores = []
        for i, network in enumerate(self.population):
            fitness = self.evaluate_fitness(network)
            self.fitness_scores.append(fitness)
            print(f"Network {i+1}/{len(self.population)}: {fitness:.2f}")
        
        # Track best network
        best_idx = np.argmax(self.fitness_scores)
        if self.fitness_scores[best_idx] > self.best_fitness:
            self.best_fitness = self.fitness_scores[best_idx]
            self.best_network = self.population[best_idx].copy()
        
        print(f"Generation {self.generation} complete!")
        print(f"Best fitness: {self.best_fitness:.2f}")
        print(f"Average fitness: {np.mean(self.fitness_scores):.2f}")
        print("-" * 50)
        
        # Create new population
        new_population = []
        
        # Keep best network (elitism)
        new_population.append(self.best_network.copy())
        
        # Generate rest of population
        while len(new_population) < self.population_size:
            parent1, parent2 = self.select_parents(self.fitness_scores)
            child = self.crossover(parent1, parent2)
            child = self.mutate(child)
            new_population.append(child)
        
        self.population = new_population
        self.generation += 1

def train_snake_ai(generations=50):
    """Train the snake AI using genetic algorithm"""
    ga = GeneticAlgorithm()
    
    for gen in range(generations):
        ga.evolve()
        
        # Save best network periodically
        if gen % 10 == 0:
            print(f"Saving best network at generation {gen}")
            # You can save the network weights here if needed
    
    return ga.best_network

def play_game_with_ai(network, visual=True):
    """Play a game with the trained AI"""
    if visual:
        pygame.init()
        screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("Snake AI")
        clock = pygame.time.Clock()
    
    game = Game()
    
    while not game.game_over:
        if visual:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
        
        state = game.get_state()
        action = network.predict(state)
        game.step(action)
        
        if visual:
            # Draw game
            screen.fill(BLACK)
            
            # Draw snake
            for segment in game.snake.body:
                rect = pygame.Rect(segment[0] * CELL_SIZE, segment[1] * CELL_SIZE, 
                                 CELL_SIZE, CELL_SIZE)
                pygame.draw.rect(screen, GREEN, rect)
            
            # Draw food
            food_rect = pygame.Rect(game.food[0] * CELL_SIZE, game.food[1] * CELL_SIZE, 
                                  CELL_SIZE, CELL_SIZE)
            pygame.draw.rect(screen, RED, food_rect)
            
            pygame.display.flip()
            clock.tick(10)  # 10 FPS
    
    if visual:
        pygame.quit()
    
    print(f"Game over! Score: {game.snake.score}, Steps: {game.snake.steps}")
    return game.snake.score

if __name__ == "__main__":
    print("Training Snake AI with Genetic Algorithm...")
    print("This may take a while...")
    
    # Train the AI
    best_network = train_snake_ai(generations=30)
    
    # Test the trained AI
    print("\nTesting trained AI:")
    for i in range(5):
        score = play_game_with_ai(best_network, visual=False)
        print(f"Test game {i+1}: Score {score}")
    
    # Play visual game
    print("\nPlaying visual game with trained AI...")
    play_game_with_ai(best_network, visual=True)
