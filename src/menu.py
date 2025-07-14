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
        button_height = 55
        button_width = 320
        # Position buttons in lower third of screen for better spacing
        start_y = gl.WINDOW_HEIGHT // 3 + 80
        
        for i in range(len(self.options)):
            button_rect = pygame.Rect(
                gl.WINDOW_WIDTH // 2 - button_width // 2,
                start_y + i * (button_height + 15),
                button_width,
                button_height
            )
            self.button_rects.append(button_rect)
    
    def draw(self):
        """Draw the menu with beautiful, well-spaced styling"""
        # Clean background
        self.screen.fill(gl.BACKGROUND)
        
        # Subtle animated background pattern
        import time
        offset = int(time.time() * 50) % 80
        for i in range(-40, gl.WINDOW_WIDTH + 40, 80):
            for j in range(-40, gl.WINDOW_HEIGHT + 40, 80):
                alpha = 8
                dot_surface = pygame.Surface((3, 3), pygame.SRCALPHA)
                dot_surface.fill((*gl.GRID_COLOR, alpha))
                self.screen.blit(dot_surface, (i + offset//2, j + offset//3))
        
        # Main title section - upper third
        title_y = 60
        title_text = self.font_title.render("SNAKE AI", True, gl.ACCENT_COLOR)
        title_rect = title_text.get_rect(centerx=gl.WINDOW_WIDTH // 2, y=title_y)
        
        # Subtle title glow
        for offset in range(2, 0, -1):
            glow_color = (*gl.ACCENT_COLOR, 30//offset)
            glow_text = self.font_title.render("SNAKE AI", True, glow_color)
            glow_rect = title_text.get_rect(centerx=gl.WINDOW_WIDTH // 2, y=title_y)
            glow_rect.x -= offset
            glow_rect.y -= offset
            self.screen.blit(glow_text, glow_rect)
        
        self.screen.blit(title_text, title_rect)
        
        # Subtitle
        subtitle_y = title_y + 60
        subtitle_text = self.font_small.render("Genetic Algorithm Neural Network", True, gl.TEXT_SECONDARY)
        subtitle_rect = subtitle_text.get_rect(centerx=gl.WINDOW_WIDTH // 2, y=subtitle_y)
        self.screen.blit(subtitle_text, subtitle_rect)
        
        # Decorative line separator
        line_y = subtitle_y + 35
        line_width = 200
        line_start = gl.WINDOW_WIDTH // 2 - line_width // 2
        line_end = gl.WINDOW_WIDTH // 2 + line_width // 2
        
        # Gradient line effect
        for i in range(line_width):
            alpha = int(255 * (1 - abs(i - line_width//2) / (line_width//2)) * 0.6)
            if alpha > 0:
                pygame.draw.line(self.screen, (*gl.ACCENT_COLOR, alpha), 
                               (line_start + i, line_y), (line_start + i, line_y + 2))
        
        # Snake decoration - positioned safely above buttons
        snake_y = line_y + 25
        snake_segments = 5
        segment_spacing = 15
        start_x = gl.WINDOW_WIDTH // 2 - (snake_segments * segment_spacing) // 2
        
        for i in range(snake_segments):
            x = start_x + i * segment_spacing
            size = 6 if i == 0 else 4  # Head is larger
            color = gl.SNAKE_HEAD if i == 0 else gl.SNAKE_BODY
            
            pygame.draw.circle(self.screen, color, (x, snake_y), size)
            if i == 0:  # Add eyes to head
                pygame.draw.circle(self.screen, gl.BACKGROUND, (x-2, snake_y-1), 1)
                pygame.draw.circle(self.screen, gl.BACKGROUND, (x+2, snake_y-1), 1)
        
        # Menu buttons section - properly spaced
        for i, (option, rect) in enumerate(zip(self.options, self.button_rects)):
            is_disabled = (i == 1 and not self.model_exists)
            is_selected = (i == self.selected_option)
            
            # Button styling
            if is_disabled:
                button_color = gl.GRID_COLOR
                border_color = gl.TEXT_SECONDARY
                text_color = gl.TEXT_SECONDARY
                shadow_offset = 1
            elif is_selected:
                button_color = gl.ACCENT_COLOR
                border_color = gl.SNAKE_HEAD
                text_color = gl.BACKGROUND
                shadow_offset = 3
            else:
                button_color = gl.BUTTON_COLOR
                border_color = gl.TEXT_COLOR
                text_color = gl.TEXT_COLOR
                shadow_offset = 2
            
            # Button shadow
            shadow_rect = pygame.Rect(rect.x + shadow_offset, rect.y + shadow_offset, 
                                    rect.width, rect.height)
            shadow_color = (15, 15, 20, 100) if is_selected else (20, 20, 25, 80)
            shadow_surface = pygame.Surface((rect.width, rect.height), pygame.SRCALPHA)
            shadow_surface.fill(shadow_color)
            self.screen.blit(shadow_surface, (shadow_rect.x, shadow_rect.y))
            
            # Main button
            pygame.draw.rect(self.screen, button_color, rect, border_radius=8)
            pygame.draw.rect(self.screen, border_color, rect, 2, border_radius=8)
            
            # Inner highlight for selected button
            if is_selected and not is_disabled:
                inner_rect = pygame.Rect(rect.x + 3, rect.y + 3, rect.width - 6, rect.height - 6)
                highlight_surface = pygame.Surface((inner_rect.width, inner_rect.height), pygame.SRCALPHA)
                highlight_surface.fill((*gl.ACCENT_COLOR, 40))
                self.screen.blit(highlight_surface, (inner_rect.x, inner_rect.y))
                pygame.draw.rect(self.screen, (*gl.SNAKE_HEAD, 100), inner_rect, 1, border_radius=6)
            
            # Button text - properly centered
            button_text = self.font_button.render(option, True, text_color)
            text_rect = button_text.get_rect(center=(rect.centerx - 15, rect.centery))  # Offset for icon space
            self.screen.blit(button_text, text_rect)
            
            # Button icons - positioned to not overlap text
            if not is_disabled:
                icon_x = rect.right - 20
                icon_y = rect.centery
                icon_color = gl.BACKGROUND if is_selected else gl.ACCENT_COLOR
                
                if i == 0:  # New Game - AI brain icon
                    pygame.draw.circle(self.screen, icon_color, (icon_x, icon_y), 7, 2)
                    pygame.draw.circle(self.screen, icon_color, (icon_x-3, icon_y-2), 2)
                    pygame.draw.circle(self.screen, icon_color, (icon_x+3, icon_y-2), 2)
                    pygame.draw.circle(self.screen, icon_color, (icon_x, icon_y+3), 2)
                elif i == 1:  # Load Model - play triangle
                    points = [(icon_x-4, icon_y-5), (icon_x-4, icon_y+5), (icon_x+5, icon_y)]
                    pygame.draw.polygon(self.screen, icon_color, points)
                elif i == 2:  # Quit - X
                    pygame.draw.line(self.screen, icon_color, 
                                   (icon_x-4, icon_y-4), (icon_x+4, icon_y+4), 2)
                    pygame.draw.line(self.screen, icon_color, 
                                   (icon_x-4, icon_y+4), (icon_x+4, icon_y-4), 2)
        
        # Footer with controls - positioned at bottom
        footer_y = gl.WINDOW_HEIGHT - 30
        controls = "UP/DOWN Navigate • Enter Select • Esc Quit"
        footer_text = self.font_small.render(controls, True, gl.TEXT_SECONDARY)
        footer_rect = footer_text.get_rect(centerx=gl.WINDOW_WIDTH // 2, y=footer_y)
        self.screen.blit(footer_text, footer_rect)
        
        # Model status indicator
        if not self.model_exists:
            status_y = footer_y - 25
            status_text = self.font_small.render("No saved model found", True, (150, 100, 100))
            status_rect = status_text.get_rect(centerx=gl.WINDOW_WIDTH // 2, y=status_y)
            self.screen.blit(status_text, status_rect)
        
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
