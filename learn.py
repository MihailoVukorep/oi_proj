#!/usr/bin/env python

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import pygame
import src.globals as gl
from src.genetic import GeneticAlgorithm 
from src.game import Game


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
        screen = pygame.display.set_mode((gl.WINDOW_WIDTH, gl.WINDOW_HEIGHT))
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
            screen.fill(gl.BLACK)
            
            # Draw snake
            for segment in game.snake.body:
                rect = pygame.Rect(segment[0] * gl.CELL_SIZE, segment[1] * gl.CELL_SIZE, 
                                 gl.CELL_SIZE, gl.CELL_SIZE)
                pygame.draw.rect(screen, gl.GREEN, rect)
            
            # Draw food
            food_rect = pygame.Rect(game.food[0] * gl.CELL_SIZE, game.food[1] * gl.CELL_SIZE, 
                                  gl.CELL_SIZE, gl.CELL_SIZE)
            pygame.draw.rect(screen, gl.RED, food_rect)
            
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
    best_network = train_snake_ai(generations=100)
    
    # Test the trained AI
    print("\nTesting trained AI:")
    for i in range(100):
        score = play_game_with_ai(best_network, visual=False)
        print(f"Test game {i+1}: Score {score}")
    
    # Play visual game
    print("\nPlaying visual game with trained AI...")
    play_game_with_ai(best_network, visual=True)
