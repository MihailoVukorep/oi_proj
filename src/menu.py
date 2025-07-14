import pygame
import sys
import os
import src.globals as gl

class Menu:
    def __init__(self, best_model_path=None):
        pygame.init()
        self.screen = pygame.display.set_mode((gl.WINDOW_WIDTH, gl.WINDOW_HEIGHT))
        pygame.display.set_caption("Snake AI - Main Menu")
        self.clock = pygame.time.Clock()
        
        # Check if model exists
        self.best_model_path = best_model_path
        self.model_exists = best_model_path and os.path.exists(best_model_path)
        
        # Fonts
        self.font_title = pygame.font.Font(None, 72)
        self.font_button = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 24)
        
        # Menu options
        self.selected_option = 0
        self.options = [
            "New Game (Train AI)",
            "Load Model & Play",
            "Quit"
        ]
        
        # Button areas for mouse support
        self.button_rects = []
        self.setup_buttons()
    
    def setup_buttons(self):
        """Setup button rectangles for mouse interaction"""
        self.button_rects = []
        button_height = 60
        button_width = 300
        start_y = gl.WINDOW_HEIGHT // 2 - 50
        
        for i in range(len(self.options)):
            button_rect = pygame.Rect(
                gl.WINDOW_WIDTH // 2 - button_width // 2,
                start_y + i * (button_height + 20),
                button_width,
                button_height
            )
            self.button_rects.append(button_rect)
    
    def draw(self):
        """Draw the menu"""
        # Background
        self.screen.fill(gl.BACKGROUND)
        
        # Title
        title_text = self.font_title.render("SNAKE AI", True, gl.ACCENT_COLOR)
        title_rect = title_text.get_rect(centerx=gl.WINDOW_WIDTH // 2, y=100)
        self.screen.blit(title_text, title_rect)
        
        # Subtitle
        subtitle_text = self.font_small.render("Genetic Algorithm Neural Network", True, gl.TEXT_SECONDARY)
        subtitle_rect = subtitle_text.get_rect(centerx=gl.WINDOW_WIDTH // 2, y=160)
        self.screen.blit(subtitle_text, subtitle_rect)
        
        # Menu options
        for i, (option, rect) in enumerate(zip(self.options, self.button_rects)):
            # Check if this is the Load Model button and model doesn't exist
            is_disabled = (i == 1 and not self.model_exists)  # Index 1 is "Load Model & Play"
            
            # Button background
            if is_disabled:
                # Disabled button colors
                button_color = gl.GRID_COLOR  # Darker grey
                border_color = gl.TEXT_SECONDARY  # Lighter grey
            else:
                button_color = gl.ACCENT_COLOR if i == self.selected_option else gl.BUTTON_COLOR
                border_color = gl.TEXT_COLOR
            
            pygame.draw.rect(self.screen, button_color, rect)
            pygame.draw.rect(self.screen, border_color, rect, 2)
            
            # Button text
            if is_disabled:
                text_color = gl.TEXT_SECONDARY  # Grey text

                #display_text = option + " (No Model Found)"
                display_text = option
            else:
                text_color = gl.BACKGROUND if i == self.selected_option else gl.TEXT_COLOR
                display_text = option
            
            button_text = self.font_button.render(display_text, True, text_color)
            text_rect = button_text.get_rect(center=rect.center)
            self.screen.blit(button_text, text_rect)
        
        pygame.display.flip()
    
    def handle_events(self):
        """Handle menu events"""
        mouse_pos = pygame.mouse.get_pos()
        
        # Check mouse hover (but skip disabled options)
        for i, rect in enumerate(self.button_rects):
            if rect.collidepoint(mouse_pos):
                # Don't allow hovering over disabled Load Model button
                if not (i == 1 and not self.model_exists):
                    self.selected_option = i
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return "quit"
            
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return "quit"
                elif event.key == pygame.K_UP:
                    self._move_selection(-1)
                elif event.key == pygame.K_DOWN:
                    self._move_selection(1)
                elif event.key == pygame.K_RETURN:
                    return self.get_selected_action()
            
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # Left click
                    for i, rect in enumerate(self.button_rects):
                        if rect.collidepoint(mouse_pos):
                            # Don't allow clicking disabled Load Model button
                            if not (i == 1 and not self.model_exists):
                                self.selected_option = i
                                return self.get_selected_action()
        
        return None
    
    def _move_selection(self, direction):
        """Move selection up or down, skipping disabled options"""
        new_selection = self.selected_option
        
        for _ in range(len(self.options)):
            new_selection = (new_selection + direction) % len(self.options)
            # Skip disabled Load Model option
            if not (new_selection == 1 and not self.model_exists):
                self.selected_option = new_selection
                break
    
    def get_selected_action(self):
        """Get the action for the selected option"""
        # Don't allow selecting disabled Load Model option
        if self.selected_option == 1 and not self.model_exists:
            return None
            
        if self.selected_option == 0:
            return "new_game"
        elif self.selected_option == 1:
            return "load_model"
        elif self.selected_option == 2:
            return "quit"
    
    def run(self):
        """Run the menu loop"""
        while True:
            action = self.handle_events()
            
            if action:
                pygame.quit()
                return action
            
            self.draw()
            self.clock.tick(60)
