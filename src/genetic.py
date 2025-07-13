import random
from .game import Game
from .network import NeuralNetwork
import numpy as np

class GeneticAlgorithm:
    def __init__(self, population_size=200, mutation_rate=0.1, mutation_strength=0.1):
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
            
            while not game.game_over and steps < 1500:  # Max steps per game
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
            
            # Penalty for eating tail (self-collision)
            if game.game_over and game.snake.check_self_collision():
                fitness -= 30 
            
            # Bonus for higher scores (exponential reward)
            if score > 5:
                fitness += score * 20
            
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
