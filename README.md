# 🐍 Snake AI - Genetic Algorithm Neural Network

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://python.org)
[![Pygame](https://img.shields.io/badge/pygame-2.0%2B-red.svg)](https://pygame.org)
[![NumPy](https://img.shields.io/badge/numpy-1.19%2B-orange.svg)](https://numpy.org)

## 🎯 Overview

This project implements an AI-powered Snake game where neural networks learn to play through evolutionary algorithms. The AI starts with random behavior and gradually evolves over generations to become an expert Snake player, demonstrating the power of genetic algorithms and neural networks in game AI.

### 🌟 Key Features

- **🧠 Neural Network AI**: Feed-forward neural network that learns game strategies
- **🧬 Genetic Algorithm**: Evolutionary approach for training without traditional backpropagation
- **🎮 Interactive Gameplay**: Beautiful modern UI with real-time visualization
- **📊 Performance Tracking**: Detailed fitness metrics and generation statistics
- **💾 Model Persistence**: Save and load trained models for future use
- **🎨 Modern Design**: Clean, responsive interface with smooth animations
- **⚡ Configurable Speed**: Adjustable game speed for training and demonstration

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/MihailoVukorep/oi_proj.git
   cd oi_proj
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   python learn.py
   ```

## 🎮 How to Use

### Training a New AI Model

1. Launch the application: `python learn.py`
2. Select **"New Game (Train AI)"** from the main menu
3. Watch as the genetic algorithm evolves the neural network for "X" generations
4. The best performing model will be automatically saved
5. After training, observe the AI play in real-time visualization

### Loading and Testing Existing Models

1. Select **"Load Model & Play"** from the main menu
2. The AI will demonstrate its learned behavior
3. Use controls to interact with the demonstration

### Controls

| Key | Action |
|-----|--------|
| `ESC` | Quit application |
| `SPACE` | Pause/Resume game |
| `R` | Restart current game |
| `M` | Return to main menu |
| `S` | Toggle game speed (30/60/120/240 FPS) |
| `T` | Train new model (when in game over screen) |

## 🏗️ Architecture

### Project Structure

```
snake-ai-genetic/
├── src/
│   ├── game.py          # Core game logic and state management
│   ├── snake.py         # Snake entity with movement and collision detection
│   ├── network.py       # Neural network implementation
│   ├── genetic.py       # Genetic algorithm for evolution
│   ├── ui.py            # User interface and rendering
│   ├── menu.py          # Main menu system
│   └── globals.py       # Game constants and configuration
├── models/
│   └── best_model.json  # Saved neural network weights
├── learn.py             # Main application entry point
├── requirements.txt     # Python dependencies
└── README.md            # Project documentation
```

### Neural Network Architecture

```
Input Layer (11 neurons)
    ├── Danger Detection (3): straight, left, right
    ├── Current Direction (4): up, down, left, right
    └── Food Location (4): up, down, left, right relative to head
              ↓
Hidden Layer (16 neurons with tanh activation)
              ↓
Output Layer (4 neurons)
    └── Actions: up, down, left, right
```

### Genetic Algorithm Parameters

| Parameter | Description |
|-----------|-------------|
| Population Size |Number of networks per generation |
| Mutation Rate |Probability of weight mutation |
| Mutation Strength |Magnitude of weight changes |
| Generations |Default training iterations |
| Elite Preservation |Best network always survives |

## 📊 Fitness Evaluation

The genetic algorithm uses a sophisticated fitness function that considers multiple factors:

### Primary Metrics
- **Score (Exponential)**: Exponential reward for food consumption
- **Survival Time**: Bonus for staying alive longer
- **Food Approach**: Reward for moving toward food
- **Danger Avoidance**: Bonus for avoiding walls and self-collision

### Advanced Metrics
- **Efficiency**: Steps-to-food ratio optimization
- **Exploration**: Penalty for repetitive movement patterns
- **Risk Management**: Reward for escaping dangerous situations
- **Consistency**: Bonus for stable performance across multiple games

### Fitness Formula
```python
final_fitness = exponential_score_reward + 
                survival_bonus + 
                food_approach_bonus + 
                danger_avoidance_bonus + 
                efficiency_bonus + 
                consistency_bonus
```

## 🔧 Configuration

### Game Settings (globals.py)

```python
# Grid Configuration
GRID_WIDTH = 20          # Game board width
GRID_HEIGHT = 20         # Game board height
CELL_SIZE = 25           # Pixel size of each cell

# Training Parameters
MAX_STEPS_WITHOUT_FOOD = 100  # Game over threshold
MAX_GAME_STEPS = 2500         # Maximum steps per game

# Neural Network Architecture
INPUT_SIZE = 11          # Input features
HIDDEN_SIZE = 16         # Hidden layer neurons
OUTPUT_SIZE = 4          # Possible actions
```

### Customizing Training

Modify training parameters in `learn.py`:

```python
# Adjust generation count
best_network = train_snake_ai(generations=100)

# Modify genetic algorithm parameters
ga = GeneticAlgorithm(
    population_size=150,
    mutation_rate=0.15,
    mutation_strength=0.3
)
```

## 📈 Performance Metrics

### Training Progress Tracking

The system provides detailed metrics during training:

- **Generation Statistics**: Best, average, and worst fitness per generation
- **Improvement Rate**: Fitness improvement over time
- **Convergence Analysis**: Population diversity and selection pressure
- **Performance Benchmarks**: Multi-game average scores for consistency

### Expected Results

| Generation Range | Typical Best Score | Behavior Description |
|------------------|-------------------|---------------------|
| 0-10 | 0-2 | Random movement, frequent collisions |
| 10-25 | 3-8 | Basic food seeking, some survival instinct |
| 25-40 | 8-15 | Consistent food collection, collision avoidance |
| 40-50+ | 15-25+ | Expert play, efficient pathfinding |


## 📚 Technical Deep Dive

### Genetic Algorithm Implementation

The genetic algorithm follows a standard evolutionary approach:

1. **Initialization**: Create random population of neural networks
2. **Evaluation**: Test each network's performance using fitness function
3. **Selection**: Choose parents using tournament selection
4. **Crossover**: Combine parent networks to create offspring
5. **Mutation**: Apply random changes to maintain genetic diversity
6. **Replacement**: Form new generation with elite preservation

### Neural Network Details

The feed-forward network processes game state through:

1. **State Extraction**: Convert game board to numerical features
2. **Feature Engineering**: Relative positions, danger detection, direction encoding
3. **Forward Propagation**: Transform input through hidden layers
4. **Action Selection**: Choose movement direction from output probabilities

### Game State Representation

The neural network receives 11 input features:

```python
state = [
    danger_straight,    # Collision risk ahead
    danger_left,        # Collision risk to left
    danger_right,       # Collision risk to right
    direction_up,       # Currently moving up
    direction_down,     # Currently moving down
    direction_left,     # Currently moving left
    direction_right,    # Currently moving right
    food_up,            # Food is above snake
    food_down,          # Food is below snake
    food_left,          # Food is left of snake
    food_right          # Food is right of snake
]
```
