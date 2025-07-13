#!/usr/bin/env python

import pygame
import sys
import src.globals as gl
from src.genetic import GeneticAlgorithm 
from src.game import Game
from src.ui import UI


def train_snake_ai(generations=100):
    """Train the snake AI using genetic algorithm"""
    ga = GeneticAlgorithm()
    
    for gen in range(generations):
        ga.evolve()
        
        # Save best network periodically
        if gen % 10 == 0:
            print(f"Saving best network at generation {gen}")
            # You can save the network weights here if needed
    
    return ga.best_network

def play_game_with_ai(network, visual=True, generation=None):
    """Play a game with the trained AI"""
    ui = None
    paused = False
    
    if visual:
        pygame.init()
        screen = pygame.display.set_mode((gl.WINDOW_WIDTH, gl.WINDOW_HEIGHT))
        pygame.display.set_caption("Snake AI - Beautiful Interface")
        clock = pygame.time.Clock()
        ui = UI(screen)
    
    game = Game()
    
    while not game.game_over:
        if visual:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        pygame.quit()
                        sys.exit()
                    elif event.key == pygame.K_SPACE:
                        paused = not paused
                    elif event.key == pygame.K_r:
                        game.reset()
            
            if paused:
                clock.tick(10)
                continue
        
        state = game.get_state()
        action = network.predict(state)
        game.step(action)
        
        if visual:
            # Clear screen with background color
            screen.fill(gl.BACKGROUND)
            
            # Draw grid
            ui.draw_grid()
            
            # Draw snake
            ui.draw_snake(game.snake)
            
            # Draw food
            ui.draw_food(game.food)
            
            # Draw UI panel
            ui.draw_ui_panel(
                current_score=game.snake.score,
                generation=generation,
                steps=game.snake.steps
            )
            
            # Draw game over screen if needed
            if game.game_over:
                ui.save_high_score(game.snake.score)
                ui.draw_game_over(game.snake.score)
            
            pygame.display.flip()
            clock.tick(15)  # Slightly faster for better visual experience
    
    # If visual mode and game is over, wait for user input
    if visual and game.game_over:
        waiting_for_input = True
        while waiting_for_input:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        pygame.quit()
                        sys.exit()
                    elif event.key == pygame.K_r:
                        # Restart the game
                        game.reset()
                        return play_game_with_ai(network, visual=True, generation=generation)
                    elif event.key == pygame.K_t:
                        # Train again
                        pygame.quit()
                        return "TRAIN_AGAIN"
                    elif event.key == pygame.K_SPACE or event.key == pygame.K_RETURN:
                        # Continue to exit
                        waiting_for_input = False
            
            # Keep drawing the game over screen
            screen.fill(gl.BACKGROUND)
            ui.draw_grid()
            ui.draw_snake(game.snake)
            ui.draw_food(game.food)
            ui.draw_ui_panel(
                current_score=game.snake.score,
                generation=generation,
                steps=game.snake.steps
            )
            ui.draw_game_over(game.snake.score)
            pygame.display.flip()
            clock.tick(10)
        
        pygame.quit()
    
    print(f"Game over! Score: {game.snake.score}, Steps: {game.snake.steps}")
    return game.snake.score

if __name__ == "__main__":
    while True:
        print("Training Snake AI with Genetic Algorithm...")
        print("This may take a while...")
        
        # Train the AI
        best_network = train_snake_ai(generations=100)
        
        # Test the trained AI
        print("\nTesting trained AI:")
        scores = []
        for i in range(10):  # Reduced from 100 for faster testing
            score = play_game_with_ai(best_network, visual=False)
            scores.append(score)
            print(f"Test game {i+1}: Score {score}")
        
        avg_score = sum(scores) / len(scores)
        max_score = max(scores)
        print(f"\nAverage Score: {avg_score:.2f}")
        print(f"Max Score: {max_score}")
        
        # Play visual game
        print("\nPlaying visual game with trained AI...")
        print("Controls: ESC to quit, SPACE to pause, R to restart, T to train again")
        result = play_game_with_ai(best_network, visual=True)
        
        # Check if user wants to train again
        if result != "TRAIN_AGAIN":
            break
        
        print("\n" + "="*50)
        print("Starting new training session...")
        print("="*50 + "\n")
