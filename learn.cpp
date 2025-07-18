#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <sstream>
#include <chrono>
#include <memory>
#include <iomanip>

// JSON library (nlohmann/json) - single header implementation
#include <nlohmann/json.hpp>
using json = nlohmann::json;

// Game constants
const int GRID_WIDTH = 35;
const int GRID_HEIGHT = 35;

// Directions
enum Direction { UP = 0, DOWN = 1, LEFT = 2, RIGHT = 3 };
const std::vector<std::pair<int, int>> DIRECTION_VECTORS = {
    {0, -1}, {0, 1}, {-1, 0}, {1, 0}  // UP, DOWN, LEFT, RIGHT
};

class Snake {
private:
    std::vector<std::pair<int, int>> body;
    Direction direction;
    bool grow;
    int score;
    int steps;
    int steps_since_food;

public:
    bool alive;

    Snake() {
        reset();
    }

    void reset() {
        body.clear();
        body.push_back({GRID_WIDTH / 2, GRID_HEIGHT / 2});
        direction = RIGHT;
        grow = false;
        alive = true;
        score = 0;
        steps = 0;
        steps_since_food = 0;
    }

    void move() {
        if (!alive) return;

        auto head = body[0];
        auto dir_vec = DIRECTION_VECTORS[direction];
        std::pair<int, int> new_head = {head.first + dir_vec.first, head.second + dir_vec.second};

        // Check wall collision
        if (new_head.first < 0 || new_head.first >= GRID_WIDTH || 
            new_head.second < 0 || new_head.second >= GRID_HEIGHT) {
            alive = false;
            return;
        }

        // Check self collision
        for (const auto& segment : body) {
            if (new_head == segment) {
                alive = false;
                return;
            }
        }

        body.insert(body.begin(), new_head);

        if (!grow) {
            body.pop_back();
        } else {
            grow = false;
            score++;
            steps_since_food = 0;
        }

        steps++;
        steps_since_food++;

        // Die if taking too long without food
        //if (steps_since_food > 400) {
        //    alive = false;
        //}
    }

    void set_direction(Direction new_direction) {
        // Prevent moving backwards
        if ((new_direction == UP && direction != DOWN) ||
            (new_direction == DOWN && direction != UP) ||
            (new_direction == LEFT && direction != RIGHT) ||
            (new_direction == RIGHT && direction != LEFT)) {
            direction = new_direction;
        }
    }

    void eat_food() {
        grow = true;
    }

    const std::vector<std::pair<int, int>>& get_body() const { return body; }
    Direction get_direction() const { return direction; }
    int get_score() const { return score; }
    int get_steps() const { return steps; }
    bool check_self_collision() const {
        auto head = body[0];
        for (size_t i = 1; i < body.size(); i++) {
            if (head == body[i]) return true;
        }
        return false;
    }
};

class Game {
private:
    Snake snake;
    std::pair<int, int> food;
    bool game_over;
    std::mt19937 rng;

public:
    Game() : rng(std::chrono::steady_clock::now().time_since_epoch().count()) {
        reset();
    }

    void reset() {
        snake.reset();
        place_food();
        game_over = false;
    }

    void place_food() {
        std::uniform_int_distribution<int> x_dist(0, GRID_WIDTH - 1);
        std::uniform_int_distribution<int> y_dist(0, GRID_HEIGHT - 1);
        
        while (true) {
            int x = x_dist(rng);
            int y = y_dist(rng);
            
            bool on_snake = false;
            for (const auto& segment : snake.get_body()) {
                if (segment.first == x && segment.second == y) {
                    on_snake = true;
                    break;
                }
            }
            
            if (!on_snake) {
                food = {x, y};
                break;
            }
        }
    }

    void step(int action) {
        if (game_over) return;

        Direction new_direction = static_cast<Direction>(action);
        snake.set_direction(new_direction);
        snake.move();

        // Check food collision
        if (snake.alive && snake.get_body()[0] == food) {
            snake.eat_food();
            place_food();
        }

        if (!snake.alive) {
            game_over = true;
        }
    }

    std::vector<float> get_state() const {
        auto head = snake.get_body()[0];
        int head_x = head.first;
        int head_y = head.second;
        int food_x = food.first;
        int food_y = food.second;

        // Danger detection
        bool danger_straight = is_collision(head_x, head_y, snake.get_direction());
        
        Direction current_dir = snake.get_direction();
        Direction left_dir = static_cast<Direction>((current_dir + 3) % 4);
        Direction right_dir = static_cast<Direction>((current_dir + 1) % 4);
        
        bool danger_left = is_collision(head_x, head_y, left_dir);
        bool danger_right = is_collision(head_x, head_y, right_dir);

        // Current direction as one-hot
        bool dir_up = (snake.get_direction() == UP);
        bool dir_down = (snake.get_direction() == DOWN);
        bool dir_left = (snake.get_direction() == LEFT);
        bool dir_right = (snake.get_direction() == RIGHT);

        // Food direction relative to snake
        bool food_up = (food_y < head_y);
        bool food_down = (food_y > head_y);
        bool food_left = (food_x < head_x);
        bool food_right = (food_x > head_x);

        return {
            static_cast<float>(danger_straight),
            static_cast<float>(danger_left),
            static_cast<float>(danger_right),
            static_cast<float>(dir_up),
            static_cast<float>(dir_down),
            static_cast<float>(dir_left),
            static_cast<float>(dir_right),
            static_cast<float>(food_up),
            static_cast<float>(food_down),
            static_cast<float>(food_left),
            static_cast<float>(food_right)
        };
    }

    bool is_collision(int x, int y, Direction direction) const {
        auto dir_vec = DIRECTION_VECTORS[direction];
        int new_x = x + dir_vec.first;
        int new_y = y + dir_vec.second;

        // Wall collision
        if (new_x < 0 || new_x >= GRID_WIDTH || new_y < 0 || new_y >= GRID_HEIGHT) {
            return true;
        }

        // Self collision
        for (const auto& segment : snake.get_body()) {
            if (segment.first == new_x && segment.second == new_y) {
                return true;
            }
        }

        return false;
    }

    bool is_game_over() const { return game_over; }
    const Snake& get_snake() const { return snake; }
    const std::pair<int, int>& get_food() const { return food; }
};

class NeuralNetwork {
private:
    int input_size;
    int hidden_size;
    int output_size;
    
    std::vector<std::vector<float>> weights1;
    std::vector<float> bias1;
    std::vector<std::vector<float>> weights2;
    std::vector<float> bias2;
    
    std::mt19937 rng;

public:
    NeuralNetwork(int input_size = 11, int hidden_size = 16, int output_size = 4) 
        : input_size(input_size), hidden_size(hidden_size), output_size(output_size),
          rng(std::chrono::steady_clock::now().time_since_epoch().count()) {
        
        initialize_weights();
    }

    void initialize_weights() {
        std::normal_distribution<float> dist(0.0f, 0.5f);
        
        weights1.resize(input_size, std::vector<float>(hidden_size));
        bias1.resize(hidden_size);
        weights2.resize(hidden_size, std::vector<float>(output_size));
        bias2.resize(output_size);
        
        for (int i = 0; i < input_size; i++) {
            for (int j = 0; j < hidden_size; j++) {
                weights1[i][j] = dist(rng);
            }
        }
        
        for (int i = 0; i < hidden_size; i++) {
            bias1[i] = dist(rng);
        }
        
        for (int i = 0; i < hidden_size; i++) {
            for (int j = 0; j < output_size; j++) {
                weights2[i][j] = dist(rng);
            }
        }
        
        for (int i = 0; i < output_size; i++) {
            bias2[i] = dist(rng);
        }
    }

    std::vector<float> forward(const std::vector<float>& input) const {
        // First layer
        std::vector<float> z1(hidden_size, 0.0f);
        for (int i = 0; i < hidden_size; i++) {
            for (int j = 0; j < input_size; j++) {
                z1[i] += input[j] * weights1[j][i];
            }
            z1[i] += bias1[i];
            z1[i] = std::tanh(z1[i]);  // Activation function
        }

        // Second layer
        std::vector<float> z2(output_size, 0.0f);
        for (int i = 0; i < output_size; i++) {
            for (int j = 0; j < hidden_size; j++) {
                z2[i] += z1[j] * weights2[j][i];
            }
            z2[i] += bias2[i];
        }

        return z2;
    }

    int predict(const std::vector<float>& input) const {
        auto output = forward(input);
        return std::max_element(output.begin(), output.end()) - output.begin();
    }

    std::vector<float> get_weights() const {
        std::vector<float> weights;
        
        for (int i = 0; i < input_size; i++) {
            for (int j = 0; j < hidden_size; j++) {
                weights.push_back(weights1[i][j]);
            }
        }
        
        for (float b : bias1) {
            weights.push_back(b);
        }
        
        for (int i = 0; i < hidden_size; i++) {
            for (int j = 0; j < output_size; j++) {
                weights.push_back(weights2[i][j]);
            }
        }
        
        for (float b : bias2) {
            weights.push_back(b);
        }
        
        return weights;
    }

    void set_weights(const std::vector<float>& weights) {
        int idx = 0;
        
        // First layer weights
        for (int i = 0; i < input_size; i++) {
            for (int j = 0; j < hidden_size; j++) {
                weights1[i][j] = weights[idx++];
            }
        }
        
        // First layer bias
        for (int i = 0; i < hidden_size; i++) {
            bias1[i] = weights[idx++];
        }
        
        // Second layer weights
        for (int i = 0; i < hidden_size; i++) {
            for (int j = 0; j < output_size; j++) {
                weights2[i][j] = weights[idx++];
            }
        }
        
        // Second layer bias
        for (int i = 0; i < output_size; i++) {
            bias2[i] = weights[idx++];
        }
    }

    std::unique_ptr<NeuralNetwork> copy() const {
        auto new_net = std::make_unique<NeuralNetwork>(input_size, hidden_size, output_size);
        new_net->set_weights(get_weights());
        return new_net;
    }

    void save_model(const std::string& filepath) const {
        json model_data;
        model_data["input_size"] = input_size;
        model_data["hidden_size"] = hidden_size;
        model_data["output_size"] = output_size;
        model_data["weights"] = get_weights();
        
        std::ofstream file(filepath);
        file << model_data.dump(4);
        file.close();
        
        std::cout << "Model saved to " << filepath << std::endl;
    }

    static std::unique_ptr<NeuralNetwork> load_model(const std::string& filepath) {
        std::ifstream file(filepath);
        if (!file.is_open()) {
            throw std::runtime_error("Model file not found: " + filepath);
        }
        
        json model_data;
        file >> model_data;
        file.close();
        
        auto network = std::make_unique<NeuralNetwork>(
            model_data["input_size"],
            model_data["hidden_size"],
            model_data["output_size"]
        );
        
        std::vector<float> weights = model_data["weights"];
        network->set_weights(weights);
        
        std::cout << "Model loaded from " << filepath << std::endl;
        return network;
    }
};

class GeneticAlgorithm {
private:
    long unsigned int population_size;
    float mutation_rate;
    float mutation_strength;
    int generation;
    
    std::vector<std::unique_ptr<NeuralNetwork>> population;
    std::vector<float> fitness_scores;
    float best_fitness;
    std::unique_ptr<NeuralNetwork> best_network;
    
    std::mt19937 rng;

public:
    GeneticAlgorithm(int pop_size = 100, float mut_rate = 0.2f, float mut_strength = 0.2f)
        : population_size(pop_size), mutation_rate(mut_rate), mutation_strength(mut_strength),
          generation(0), best_fitness(0), rng(std::chrono::steady_clock::now().time_since_epoch().count()) {
        
        // Create initial population
        population.reserve(population_size);
        for (long unsigned int i = 0; i < population_size; i++) {
            population.push_back(std::make_unique<NeuralNetwork>());
        }
        
        fitness_scores.resize(population_size);
    }

    float evaluate_fitness(const NeuralNetwork& network, int games = 5) {
        float total_score = 0.0f;
        int total_steps = 0;
        int total_food_eaten = 0;
        int successful_games = 0;
        
        for (int game_num = 0; game_num < games; game_num++) {
            Game game;
            int steps = 0;
            int food_eaten = 0;
            int steps_without_food = 0;
            int moves_towards_food = 0;
            int moves_away_from_food = 0;
            int near_death_escapes = 0;
            
            std::vector<std::pair<int, int>> position_history;
            int loop_penalty = 0;
            
            while (!game.is_game_over() && steps < GRID_WIDTH * GRID_HEIGHT) {
                auto state = game.get_state();
                auto head_pos = game.get_snake().get_body()[0];
                auto food_pos = game.get_food();
                
                // Calculate distance to food before move
                int current_food_distance = abs(head_pos.first - food_pos.first) + abs(head_pos.second - food_pos.second);
                
                // Check for potential danger
                float danger_level = calculate_danger_level(game);
                
                int action = network.predict(state);
                int previous_score = game.get_snake().get_score();
                
                // Execute action
                game.step(action);
                steps++;
                
                // Track position history for loop detection
                bool found_in_recent = false;
                int recent_limit = std::min(10, (int)position_history.size());
                for (long unsigned int i = position_history.size() - recent_limit; i < position_history.size(); i++) {
                    if (position_history[i] == head_pos) {
                        found_in_recent = true;
                        break;
                    }
                }
                if (found_in_recent) {
                    loop_penalty++;
                }
                position_history.push_back(head_pos);
                
                // Check if food was eaten
                if (game.get_snake().get_score() > previous_score) {
                    food_eaten++;
                    steps_without_food = 0;
                    moves_towards_food += 5;
                } else {
                    steps_without_food++;
                    
                    // Check if moving towards or away from food
                    if (!game.is_game_over()) {
                        auto new_head_pos = game.get_snake().get_body()[0];
                        int new_food_distance = abs(new_head_pos.first - food_pos.first) + abs(new_head_pos.second - food_pos.second);
                        
                        if (new_food_distance < current_food_distance) {
                            moves_towards_food++;
                        } else if (new_food_distance > current_food_distance) {
                            moves_away_from_food++;
                        }
                    }
                }
                
                // Check for near-death escapes
                if (danger_level > 0.7f && !game.is_game_over()) {
                    near_death_escapes++;
                }
                
                // Early termination if stuck
                if (steps_without_food > 400) {
                    break;
                }
            }
            
            // Calculate comprehensive fitness
            int score = game.get_snake().get_score();
            
            // Score fitness (exponential reward)
            float score_fitness = (score == 0) ? 0.0f : (score * score * 1000.0f);
            
            // Survival bonus
            float survival_bonus = std::min(steps * 0.8f, 800.0f);
            
            // Food-seeking behavior reward
            float food_seeking_bonus = (steps > 0) ? (float(moves_towards_food) / steps * 600.0f) : 0.0f;
            
            // Efficiency bonus
            float efficiency_bonus = (steps > 0) ? (float(food_eaten) / steps * 1500.0f) : 0.0f;
            
            // Length achievement bonus
            float length_bonus = game.get_snake().get_body().size() * 150.0f;
            
            // Progressive difficulty bonus
            float difficulty_bonus = (score > 0) ? (score * 100.0f * (1.0f + score * 0.1f)) : 0.0f;
            
            // Consistency bonus
            float consistency_bonus = (score >= 3) ? ((score - 2) * 300.0f) : 0.0f;
            
            // Near-death escape bonus
            float escape_bonus = near_death_escapes * 50.0f;
            
            // Calculate penalties
            float penalties = 0.0f;
            
            // Early death penalty
            if (score == 0 && steps < 100) {
                penalties += 800.0f;
            }
            
            // Self-collision penalty
            if (game.is_game_over() && game.get_snake().check_self_collision()) {
                if (score >= 40) penalties += 50.0f;
                else if (score >= 30) penalties += 100.0f;
                else if (score >= 20) penalties += 200.0f;
                else if (score >= 10) penalties += 350.0f;
                else if (score >= 5) penalties += 500.0f;
                else penalties += 800.0f;
            }
            
            // Loop penalty
            penalties += loop_penalty * 5.0f;
            
            // Inefficiency penalty
            if (steps > 200 && food_eaten == 0) {
                penalties += 300.0f;
            }
            
            // Moving away from food penalty
            if (moves_away_from_food > moves_towards_food && score < 3) {
                penalties += 200.0f;
            }
            
            // Calculate final game fitness
            float game_fitness = score_fitness + survival_bonus + food_seeking_bonus + 
                               efficiency_bonus + length_bonus + difficulty_bonus + 
                               consistency_bonus + escape_bonus - penalties;
            
            game_fitness = std::max(game_fitness, -1000.0f);
            
            total_score += game_fitness;
            total_steps += steps;
            total_food_eaten += food_eaten;
            
            if (score > 0) {
                successful_games++;
            }
        }
        
        // Calculate average fitness
        float avg_fitness = total_score / games;
        
        // Add consistency bonus across games
        float consistency_bonus = (float(successful_games) / games) * 800.0f;
        
        // Add total performance bonus
        float performance_bonus = (total_food_eaten > games * 2) ? 
            ((total_food_eaten - games * 2) * 200.0f) : 0.0f;
        
        return avg_fitness + consistency_bonus + performance_bonus;
    }

    float calculate_danger_level(const Game& game) const {
        auto head_pos = game.get_snake().get_body()[0];
        float danger_score = 0.0f;
        
        // Distance to each wall
        std::vector<int> wall_distances = {
            head_pos.first,  // Left wall
            head_pos.second,  // Top wall
            GRID_WIDTH - head_pos.first - 1,  // Right wall
            GRID_HEIGHT - head_pos.second - 1  // Bottom wall
        };
        
        // Add danger for being close to walls
        for (int dist : wall_distances) {
            if (dist <= 1) {
                danger_score += 0.3f;
            } else if (dist <= 2) {
                danger_score += 0.1f;
            }
        }
        
        // Check proximity to body parts
        const auto& body = game.get_snake().get_body();
        for (size_t i = 1; i < body.size(); i++) {
            int distance = abs(head_pos.first - body[i].first) + abs(head_pos.second - body[i].second);
            if (distance <= 1) {
                danger_score += 0.4f;
            } else if (distance <= 2) {
                danger_score += 0.2f;
            }
        }
        
        return std::min(danger_score, 1.0f);
    }

    std::pair<int, int> select_parents() {
        std::uniform_int_distribution<int> dist(0, population_size - 1);
        
        std::vector<int> tournament1, tournament2;
        int tournament_size = 5;
        
        for (int i = 0; i < tournament_size; i++) {
            tournament1.push_back(dist(rng));
            tournament2.push_back(dist(rng));
        }
        
        int winner1 = *std::max_element(tournament1.begin(), tournament1.end(), 
            [this](int a, int b) { return fitness_scores[a] < fitness_scores[b]; });
        int winner2 = *std::max_element(tournament2.begin(), tournament2.end(), 
            [this](int a, int b) { return fitness_scores[a] < fitness_scores[b]; });
        
        return {winner1, winner2};
    }

    std::unique_ptr<NeuralNetwork> crossover(const NeuralNetwork& parent1, const NeuralNetwork& parent2) {
        auto child = std::make_unique<NeuralNetwork>();
        
        auto weights1 = parent1.get_weights();
        auto weights2 = parent2.get_weights();
        
        std::uniform_int_distribution<int> dist(1, weights1.size() - 1);
        int crossover_point = dist(rng);
        
        std::vector<float> child_weights;
        child_weights.reserve(weights1.size());
        
        for (int i = 0; i < crossover_point; i++) {
            child_weights.push_back(weights1[i]);
        }
        for (long unsigned int i = crossover_point; i < weights2.size(); i++) {
            child_weights.push_back(weights2[i]);
        }
        
        child->set_weights(child_weights);
        return child;
    }

    void mutate(NeuralNetwork& network) {
        auto weights = network.get_weights();
        std::uniform_real_distribution<float> uniform_dist(0.0f, 1.0f);
        std::normal_distribution<float> normal_dist(0.0f, mutation_strength);
        
        for (float& weight : weights) {
            if (uniform_dist(rng) < mutation_rate) {
                weight += normal_dist(rng);
            }
        }
        
        network.set_weights(weights);
    }

    void evolve() {
        std::cout << "Generation " << generation << ": Evaluating fitness..." << std::endl;
        
        // Evaluate fitness for all networks
        for (long unsigned int i = 0; i < population_size; i++) {
            fitness_scores[i] = evaluate_fitness(*population[i]);
        }
        
        // Track best network
        int best_idx = std::max_element(fitness_scores.begin(), fitness_scores.end()) - fitness_scores.begin();
        if (fitness_scores[best_idx] > best_fitness) {
            best_fitness = fitness_scores[best_idx];
            best_network = population[best_idx]->copy();
        }
        
        std::cout << "Generation " << generation << " complete!" << std::endl;
        std::cout << "Best fitness: " << std::fixed << std::setprecision(2) << best_fitness << std::endl;
        
        float avg_fitness = 0.0f;
        for (float fitness : fitness_scores) {
            avg_fitness += fitness;
        }
        avg_fitness /= population_size;
        std::cout << "Average fitness: " << std::fixed << std::setprecision(2) << avg_fitness << std::endl;
        std::cout << std::string(50, '-') << std::endl;
        
        // Create new population
        std::vector<std::unique_ptr<NeuralNetwork>> new_population;
        new_population.reserve(population_size);
        
        // Keep best network (elitism)
        new_population.push_back(best_network->copy());
        
        // Generate rest of population
        while (new_population.size() < population_size) {
            auto parents = select_parents();
            auto child = crossover(*population[parents.first], *population[parents.second]);
            mutate(*child);
            new_population.push_back(std::move(child));
        }
        
        population = std::move(new_population);
        generation++;
    }

    void save_best_model(const std::string& filepath) const {
        if (best_network) {
            best_network->save_model(filepath);
        }
    }

    float get_best_fitness() const { return best_fitness; }
    int get_generation() const { return generation; }
};

int main() {
    std::cout << "Snake AI Training in C++" << std::endl;
    std::cout << "=========================" << std::endl;
    
    int population_size = 300;

    // Initialize genetic algorithm
    GeneticAlgorithm ga(population_size, 0.2f, 0.2f);  // population_size, mutation_rate, mutation_strength
    
    // Training loop
    int max_generations = 1000;
    int save_interval = 250;  // Save model every 250 generations
    
    for (int i = 0; i < max_generations; i++) {
        auto start = std::chrono::high_resolution_clock::now();
        
        ga.evolve();
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        
        std::cout << "Generation time: " << duration.count() << "ms" << std::endl;
        
        // Save model periodically
        if ((i + 1) % save_interval == 0) {
            std::string filename = "models/snake_ai_pop_" + std::to_string(population_size) + "_gen_" + std::to_string(ga.get_generation()) + ".json";
            ga.save_best_model(filename);
        }
        
        std::cout << std::endl;
    }
    
    // Save final model
    std::string final_filename = "models/snake_ai_final.json";
    ga.save_best_model(final_filename);
    
    std::cout << "Training completed!" << std::endl;
    std::cout << "Best fitness achieved: " << ga.get_best_fitness() << std::endl;
    
    return 0;
}
