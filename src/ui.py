import pygame
import json
import os
from .globals import *

class UI:
    def __init__(self, screen):
        self.screen = screen
        self.font_large = pygame.font.Font(None, 36)
        self.font_medium = pygame.font.Font(None, 24)
        self.font_small = pygame.font.Font(None, 20)
        self.high_score = self.load_high_score()
        
    def load_high_score(self):
        """Load high score from file"""
        try:
            if os.path.exists('high_score.json'):
                with open('high_score.json', 'r') as f:
                    data = json.load(f)
                    return data.get('high_score', 0)
        except:
            pass
        return 0
    
    def save_high_score(self, score):
        """Save high score to file"""
        if score > self.high_score:
            self.high_score = score
            try:
                with open('high_score.json', 'w') as f:
                    json.dump({'high_score': score}, f)
            except:
                pass
    
    def draw_grid(self):
        """Draw the game grid with modern styling"""
        game_area = pygame.Rect(0, 0, GAME_AREA_WIDTH, GAME_AREA_HEIGHT)
        pygame.draw.rect(self.screen, BACKGROUND, game_area)
        
        # Draw grid lines
        for x in range(0, GAME_AREA_WIDTH + 1, CELL_SIZE):
            pygame.draw.line(self.screen, GRID_COLOR, (x, 0), (x, GAME_AREA_HEIGHT), 1)
        for y in range(0, GAME_AREA_HEIGHT + 1, CELL_SIZE):
            pygame.draw.line(self.screen, GRID_COLOR, (0, y), (GAME_AREA_WIDTH, y), 1)
    
    def draw_snake(self, snake):
        """Draw the snake with beautiful styling"""
        for i, segment in enumerate(snake.body):
            x, y = segment[0] * CELL_SIZE, segment[1] * CELL_SIZE
            
            if i == 0:  # Head
                # Draw head with glow effect
                head_rect = pygame.Rect(x + 2, y + 2, CELL_SIZE - 4, CELL_SIZE - 4)
                pygame.draw.rect(self.screen, SNAKE_HEAD, head_rect, border_radius=8)
                
                # Add eyes
                eye_size = 4
                eye1_pos = (x + 6, y + 6)
                eye2_pos = (x + CELL_SIZE - 10, y + 6)
                pygame.draw.circle(self.screen, BLACK, eye1_pos, eye_size)
                pygame.draw.circle(self.screen, BLACK, eye2_pos, eye_size)
            else:  # Body
                # Draw body segments with gradient effect
                body_rect = pygame.Rect(x + 1, y + 1, CELL_SIZE - 2, CELL_SIZE - 2)
                pygame.draw.rect(self.screen, SNAKE_BODY, body_rect, border_radius=6)
                
                # Add subtle shadow
                shadow_rect = pygame.Rect(x + 3, y + 3, CELL_SIZE - 6, CELL_SIZE - 6)
                pygame.draw.rect(self.screen, SNAKE_SHADOW, shadow_rect, border_radius=4)
    
    def draw_food(self, food_pos):
        """Draw the food with glow effect"""
        x, y = food_pos[0] * CELL_SIZE, food_pos[1] * CELL_SIZE
        center = (x + CELL_SIZE // 2, y + CELL_SIZE // 2)
        
        # Draw glow effect
        for radius in range(15, 8, -2):
            alpha = 30 - (radius - 8) * 3
            glow_surface = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
            pygame.draw.circle(glow_surface, (*FOOD_GLOW, alpha), (radius, radius), radius)
            self.screen.blit(glow_surface, (center[0] - radius, center[1] - radius))
        
        # Draw food
        pygame.draw.circle(self.screen, FOOD_COLOR, center, 10)
        pygame.draw.circle(self.screen, FOOD_GLOW, center, 6)
    
    def draw_ui_panel(self, current_score, generation=None, steps=None):
        """Draw the UI panel with scores and info"""
        panel_rect = pygame.Rect(GAME_AREA_WIDTH, 0, UI_WIDTH, WINDOW_HEIGHT)
        pygame.draw.rect(self.screen, PANEL_COLOR, panel_rect)
        
        # Draw vertical separator
        pygame.draw.line(self.screen, ACCENT_COLOR, 
                        (GAME_AREA_WIDTH, 0), (GAME_AREA_WIDTH, WINDOW_HEIGHT), 3)
        
        y_offset = 30
        
        # Title
        title = self.font_large.render("SNAKE AI", True, TEXT_COLOR)
        title_rect = title.get_rect(centerx=GAME_AREA_WIDTH + UI_WIDTH // 2, y=y_offset)
        self.screen.blit(title, title_rect)
        y_offset += 60
        
        # Current Score
        score_label = self.font_medium.render("CURRENT SCORE", True, TEXT_SECONDARY)
        score_rect = score_label.get_rect(centerx=GAME_AREA_WIDTH + UI_WIDTH // 2, y=y_offset)
        self.screen.blit(score_label, score_rect)
        y_offset += 30
        
        score_value = self.font_large.render(str(current_score), True, ACCENT_COLOR)
        score_value_rect = score_value.get_rect(centerx=GAME_AREA_WIDTH + UI_WIDTH // 2, y=y_offset)
        self.screen.blit(score_value, score_value_rect)
        y_offset += 60
        
        # High Score
        high_score_label = self.font_medium.render("HIGH SCORE", True, TEXT_SECONDARY)
        high_score_rect = high_score_label.get_rect(centerx=GAME_AREA_WIDTH + UI_WIDTH // 2, y=y_offset)
        self.screen.blit(high_score_label, high_score_rect)
        y_offset += 30
        
        high_score_value = self.font_large.render(str(self.high_score), True, FOOD_COLOR)
        high_score_value_rect = high_score_value.get_rect(centerx=GAME_AREA_WIDTH + UI_WIDTH // 2, y=y_offset)
        self.screen.blit(high_score_value, high_score_value_rect)
        y_offset += 60
        
        # Additional info if provided
        if generation is not None:
            gen_label = self.font_medium.render("GENERATION", True, TEXT_SECONDARY)
            gen_rect = gen_label.get_rect(centerx=GAME_AREA_WIDTH + UI_WIDTH // 2, y=y_offset)
            self.screen.blit(gen_label, gen_rect)
            y_offset += 30
            
            gen_value = self.font_medium.render(str(generation), True, TEXT_COLOR)
            gen_value_rect = gen_value.get_rect(centerx=GAME_AREA_WIDTH + UI_WIDTH // 2, y=y_offset)
            self.screen.blit(gen_value, gen_value_rect)
            y_offset += 40
        
        if steps is not None:
            steps_label = self.font_small.render("STEPS", True, TEXT_SECONDARY)
            steps_rect = steps_label.get_rect(centerx=GAME_AREA_WIDTH + UI_WIDTH // 2, y=y_offset)
            self.screen.blit(steps_label, steps_rect)
            y_offset += 25
            
            steps_value = self.font_small.render(str(steps), True, TEXT_COLOR)
            steps_value_rect = steps_value.get_rect(centerx=GAME_AREA_WIDTH + UI_WIDTH // 2, y=y_offset)
            self.screen.blit(steps_value, steps_value_rect)
            y_offset += 40
        
        # Controls info
        controls_y = WINDOW_HEIGHT - 120
        controls_title = self.font_small.render("CONTROLS", True, TEXT_SECONDARY)
        controls_title_rect = controls_title.get_rect(centerx=GAME_AREA_WIDTH + UI_WIDTH // 2, y=controls_y)
        self.screen.blit(controls_title, controls_title_rect)
        
        controls_text = [
            "ESC - Quit",
            "SPACE - Pause",
            "R - Restart"
        ]
        
        for i, text in enumerate(controls_text):
            control_text = self.font_small.render(text, True, TEXT_COLOR)
            control_rect = control_text.get_rect(centerx=GAME_AREA_WIDTH + UI_WIDTH // 2, 
                                               y=controls_y + 25 + i * 20)
            self.screen.blit(control_text, control_rect)
    
    def draw_game_over(self, final_score):
        """Draw game over screen"""
        # Semi-transparent overlay
        overlay = pygame.Surface((GAME_AREA_WIDTH, GAME_AREA_HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))
        self.screen.blit(overlay, (0, 0))
        
        # Game over text
        game_over_text = self.font_large.render("GAME OVER", True, FOOD_COLOR)
        game_over_rect = game_over_text.get_rect(centerx=GAME_AREA_WIDTH // 2, 
                                               centery=GAME_AREA_HEIGHT // 2 - 80)
        self.screen.blit(game_over_text, game_over_rect)
        
        # Final score
        score_text = self.font_medium.render(f"Final Score: {final_score}", True, TEXT_COLOR)
        score_rect = score_text.get_rect(centerx=GAME_AREA_WIDTH // 2, 
                                       centery=GAME_AREA_HEIGHT // 2 - 30)
        self.screen.blit(score_text, score_rect)
        
        # Check for new high score
        if final_score > self.high_score:
            new_high_text = self.font_medium.render("NEW HIGH SCORE!", True, ACCENT_COLOR)
            new_high_rect = new_high_text.get_rect(centerx=GAME_AREA_WIDTH // 2, 
                                                 centery=GAME_AREA_HEIGHT // 2 + 10)
            self.screen.blit(new_high_text, new_high_rect)
        
        # Instructions
        instructions = [
            "R - Restart Game",
            "T - Train Again",
            "SPACE/ENTER - Continue",
            "ESC - Quit"
        ]
        
        for i, instruction in enumerate(instructions):
            instruction_text = self.font_small.render(instruction, True, TEXT_SECONDARY)
            instruction_rect = instruction_text.get_rect(centerx=GAME_AREA_WIDTH // 2, 
                                                       centery=GAME_AREA_HEIGHT // 2 + 60 + i * 25)
            self.screen.blit(instruction_text, instruction_rect)
