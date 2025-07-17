#!/usr/bin/env python

import pygame
import sys
import os
import src.globals as gl
from src.genetic import GeneticAlgorithm 
from src.game import Game
from src.ui import UI
from src.network import NeuralNetwork
from src.menu import Menu

# Model save path
MODELS_DIR = "models"
BEST_MODEL_PATH = os.path.join(MODELS_DIR, "best_model.json")

def train_snake_ai(generations=50):
    """Train the snake AI using genetic algorithm"""
    ga = GeneticAlgorithm(population_size=300)
    
    for gen in range(generations):
        ga.evolve()
        
        # Show progress every 10 generations
        if gen % 10 == 0:
            print(f"Generation {gen}: Best fitness so far: {ga.best_fitness:.2f}")
    
    # Save only the final best model
    if ga.best_network:
        ga.best_network.save_model(BEST_MODEL_PATH)
        print(f"Best model saved to {BEST_MODEL_PATH}")
    
    return ga.best_network

def load_best_model():
    """Load the best saved model"""
    try:
        network = NeuralNetwork.load_model(BEST_MODEL_PATH)
        return network
    except FileNotFoundError:
        print(f"No saved model found at {BEST_MODEL_PATH}")
        print("Please train a model first by selecting 'New Game (Train AI)'")
        return None

def play_game_with_ai(network, visual=True, generation=None):
    """Play a game with the trained AI"""
    ui = None
    paused = False
    
    # Speed settings for S key toggle
    speed_levels = [30, 60, 120, 240]  # FPS values
    current_speed_index = 1  # Start with 60 FPS
    
    if visual:
        pygame.init()
        screen = pygame.display.set_mode((gl.WINDOW_WIDTH, gl.WINDOW_HEIGHT))
        pygame.display.set_caption("Snake AI")
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
                    elif event.key == pygame.K_m:
                        # Return to main menu
                        pygame.quit()
                        return "MAIN_MENU"
                    elif event.key == pygame.K_s:
                        # Toggle speed
                        current_speed_index = (current_speed_index + 1) % len(speed_levels)
                        print(f"Speed changed to {speed_levels[current_speed_index]} FPS")
            
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
            clock.tick(speed_levels[current_speed_index])  # Use variable speed
    
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
                    elif event.key == pygame.K_m:
                        # Return to main menu
                        pygame.quit()
                        return "MAIN_MENU"
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
        # Show main menu with model path check
        menu = Menu(BEST_MODEL_PATH)
        action = menu.run()

        if action == "quit":
            print("Goodbye!")
            break

        elif action == "new_game":
            print("Training Snake AI with Genetic Algorithm...")
            print("This may take a while...")
            
            # Train the AI
            best_network = train_snake_ai(generations=1000)
            
            if best_network is None:
                print("Training failed!")
                continue
            
            # Test the trained AI
            print("\nTesting trained AI:")
            scores = []
            for i in range(10):
                score = play_game_with_ai(best_network, visual=False)
                scores.append(score)
                print(f"Test game {i+1}: Score {score}")
            
            avg_score = sum(scores) / len(scores)
            max_score = max(scores)
            print(f"\nAverage Score: {avg_score:.2f}")
            print(f"Max Score: {max_score}")
            
            # Play visual game
            print("\nPlaying visual game with trained AI...")
            print("Controls: ESC to quit, SPACE to pause, R to restart, M to main menu, S to toggle speed, T to train again")
            result = play_game_with_ai(best_network, visual=True)
            
            # Check if user wants to train again or return to menu
            if result == "TRAIN_AGAIN":
                print("\n" + "="*50)
                print("Starting new training session...")
                print("="*50 + "\n")
                continue
            elif result == "MAIN_MENU":
                print("\nReturning to main menu...")
                continue
        
        elif action == "load_model":
            print("Loading best saved model...")
            best_network = load_best_model()
            
            if best_network is None:
                print("Press any key to return to menu...")
                input()
                continue
            
            print("Model loaded successfully!")
            print("Playing game with loaded AI...")
            print("Controls: ESC to quit, SPACE to pause, R to restart, M to main menu, S to toggle speed")
            
            # Play visual game with loaded model
            result = play_game_with_ai(best_network, visual=True)
            
            if result == "TRAIN_AGAIN":
                print("\n" + "="*50)
                print("Starting new training session...")
                print("="*50 + "\n")
                continue
            elif result == "MAIN_MENU":
                print("\nReturning to main menu...")
                continue
        
        # Ask if user wants to return to menu or exit
        print("\nReturning to main menu...")
        print("="*50)
