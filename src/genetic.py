import random
from .game import Game
from .network import NeuralNetwork
import numpy as np

class GeneticAlgorithm:
    def __init__(self, population_size=100, mutation_rate=0.2, mutation_strength=0.2):
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
    
    def evaluate_fitness(self, network, games=5):
        """Enhanced fitness evaluation with multiple sophisticated metrics"""
        total_score = 0
        total_steps = 0
        total_food_eaten = 0
        successful_games = 0
        
        for game_num in range(games):
            game = Game()
            steps = 0
            food_eaten = 0
            last_food_distance = float('inf')
            steps_without_food = 0
            moves_towards_food = 0
            moves_away_from_food = 0
            wall_collisions = 0
            near_death_escapes = 0
            
            # Track previous positions to detect loops
            position_history = []
            loop_penalty = 0
            
            while not game.game_over and steps < 2500:  # Increased max steps
                state = game.get_state()
                head_pos = game.snake.body[0]
                food_pos = game.food
                
                # Calculate distance to food before move
                current_food_distance = abs(head_pos[0] - food_pos[0]) + abs(head_pos[1] - food_pos[1])
                
                # Check for potential danger (walls and body)
                danger_level = self.calculate_danger_level(game)
                
                action = network.predict(state)
                previous_score = game.snake.score
                
                # Execute action
                game.step(action)
                steps += 1
                
                # Track position history for loop detection
                if head_pos in position_history[-10:]:  # Check last 10 positions
                    loop_penalty += 1
                position_history.append(head_pos)
                
                # Check if food was eaten
                if game.snake.score > previous_score:
                    food_eaten += 1
                    steps_without_food = 0
                    moves_towards_food += 5  # Bonus for eating
                else:
                    steps_without_food += 1
                    
                    # Check if moving towards or away from food
                    new_head_pos = game.snake.body[0] if not game.game_over else head_pos
                    new_food_distance = abs(new_head_pos[0] - food_pos[0]) + abs(new_head_pos[1] - food_pos[1])
                    
                    if new_food_distance < current_food_distance:
                        moves_towards_food += 1
                    elif new_food_distance > current_food_distance:
                        moves_away_from_food += 1
                    
                    last_food_distance = new_food_distance
                
                # Check for near-death escapes
                if danger_level > 0.7 and not game.game_over:
                    near_death_escapes += 1
                
                # Early termination if stuck in area without progress
                if steps_without_food > 400:  # Increased threshold
                    break
            
            # Calculate comprehensive fitness
            score = game.snake.score
            
            # 1. Exponential score reward (main driver)
            if score == 0:
                score_fitness = 0
            else:
                score_fitness = (score ** 2) * 1000  # Exponential reward
            
            # 2. Survival bonus (capped to prevent exploitation)
            survival_bonus = min(steps * 0.8, 800)
            
            # 3. Food-seeking behavior reward
            if steps > 0:
                food_seeking_ratio = moves_towards_food / steps
                food_seeking_bonus = food_seeking_ratio * 600
            else:
                food_seeking_bonus = 0
            
            # 4. Efficiency bonus (food per step)
            if steps > 0:
                efficiency = food_eaten / steps
                efficiency_bonus = efficiency * 1500
            else:
                efficiency_bonus = 0
            
            # 5. Length achievement bonus
            length_bonus = len(game.snake.body) * 150
            
            # 6. Progressive difficulty bonus
            if score > 0:
                difficulty_multiplier = 1 + (score * 0.1)  # Gets harder as score increases
                difficulty_bonus = score * 100 * difficulty_multiplier
            else:
                difficulty_bonus = 0
            
            # 7. Consistency bonus for multiple food items
            if score >= 3:
                consistency_bonus = (score - 2) * 300
            else:
                consistency_bonus = 0
            
            # 8. Near-death escape bonus (reward for skillful play)
            escape_bonus = near_death_escapes * 50
            
            # PENALTIES
            penalties = 0
            
            # Early death penalty
            if score == 0 and steps < 100:
                penalties += 800
            
            # Progressive self-collision penalty
            if game.game_over and hasattr(game.snake, 'check_self_collision'):
                if game.snake.check_self_collision():
                    if score >= 40:
                        penalties += 50
                    elif score >= 30:
                        penalties += 100
                    elif score >= 20:
                        penalties += 200
                    elif score >= 10:
                        penalties += 350
                    elif score >= 5:
                        penalties += 500
                    else:
                        penalties += 800
            
            # Loop penalty (discourages repetitive behavior)
            penalties += loop_penalty * 5
            
            # Inefficiency penalty (too many moves without food)
            if steps > 200 and food_eaten == 0:
                penalties += 300
            
            # Moving away from food penalty
            if moves_away_from_food > moves_towards_food and score < 3:
                penalties += 200
            
            # Calculate final game fitness
            game_fitness = (score_fitness + survival_bonus + food_seeking_bonus + 
                           efficiency_bonus + length_bonus + difficulty_bonus + 
                           consistency_bonus + escape_bonus - penalties)
            
            # Ensure minimum fitness is not too negative
            game_fitness = max(game_fitness, -1000)
            
            total_score += game_fitness
            total_steps += steps
            total_food_eaten += food_eaten
            
            if score > 0:
                successful_games += 1
        
        # Calculate average fitness
        avg_fitness = total_score / games
        
        # Add consistency bonus across games
        consistency_bonus = (successful_games / games) * 800
        
        # Add total performance bonus
        if total_food_eaten > games * 2:  # More than 2 food per game on average
            performance_bonus = (total_food_eaten - games * 2) * 200
        else:
            performance_bonus = 0
        
        final_fitness = avg_fitness + consistency_bonus + performance_bonus
        
        return final_fitness
    
    def calculate_danger_level(self, game):
        """Calculate how dangerous the current position is (0 = safe, 1 = immediate death)"""
        head_pos = game.snake.body[0]
        danger_score = 0
        
        # Check distance to walls - need to get board dimensions from globals
        from .globals import GRID_WIDTH, GRID_HEIGHT
        board_width = GRID_WIDTH
        board_height = GRID_HEIGHT
        
        # Distance to each wall (normalized)
        wall_distances = [
            head_pos[0],  # Left wall
            head_pos[1],  # Top wall
            board_width - head_pos[0] - 1,  # Right wall
            board_height - head_pos[1] - 1  # Bottom wall
        ]
        
        # Add danger for being close to walls
        for dist in wall_distances:
            if dist <= 1:
                danger_score += 0.3
            elif dist <= 2:
                danger_score += 0.1
        
        # Check proximity to body parts
        for body_part in game.snake.body[1:]:  # Skip head
            distance = abs(head_pos[0] - body_part[0]) + abs(head_pos[1] - body_part[1])
            if distance <= 1:
                danger_score += 0.4
            elif distance <= 2:
                danger_score += 0.2
        
        return min(danger_score, 1.0)  # Cap at 1.0
    
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
            #print(f"Network {i+1}/{len(self.population)}: {fitness:.2f}")
        
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
