import numpy as np
import json
import os

class NeuralNetwork:
    def __init__(self, input_size=11, hidden_size=16, output_size=4):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Initialize weights randomly
        self.weights1 = np.random.randn(input_size, hidden_size) * 0.5
        self.bias1 = np.random.randn(hidden_size) * 0.5
        self.weights2 = np.random.randn(hidden_size, output_size) * 0.5
        self.bias2 = np.random.randn(output_size) * 0.5
    
    def forward(self, x):
        """Forward pass through network"""
        # First layer
        z1 = np.dot(x, self.weights1) + self.bias1
        a1 = np.tanh(z1)  # Activation function
        
        # Second layer
        z2 = np.dot(a1, self.weights2) + self.bias2
        a2 = z2  # Linear output
        
        return a2
    
    def predict(self, x):
        """Get action from network output"""
        output = self.forward(x)
        return np.argmax(output)
    
    def get_weights(self):
        """Get all weights as a flat array"""
        return np.concatenate([
            self.weights1.flatten(),
            self.bias1.flatten(),
            self.weights2.flatten(),
            self.bias2.flatten()
        ])
    
    def set_weights(self, weights):
        """Set weights from flat array"""
        idx = 0
        
        # First layer weights
        w1_size = self.input_size * self.hidden_size
        self.weights1 = weights[idx:idx + w1_size].reshape(self.input_size, self.hidden_size)
        idx += w1_size
        
        # First layer bias
        self.bias1 = weights[idx:idx + self.hidden_size]
        idx += self.hidden_size
        
        # Second layer weights
        w2_size = self.hidden_size * self.output_size
        self.weights2 = weights[idx:idx + w2_size].reshape(self.hidden_size, self.output_size)
        idx += w2_size
        
        # Second layer bias
        self.bias2 = weights[idx:idx + self.output_size]
    
    def copy(self):
        """Create a copy of this network"""
        new_net = NeuralNetwork(self.input_size, self.hidden_size, self.output_size)
        new_net.set_weights(self.get_weights())
        return new_net
    
    def save_model(self, filepath):
        """Save the neural network weights to a file"""
        model_data = {
            'input_size': self.input_size,
            'hidden_size': self.hidden_size,
            'output_size': self.output_size,
            'weights': self.get_weights().tolist()
        }
        
        # Create models directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(model_data, f)
        print(f"Model saved to {filepath}")
    
    @classmethod
    def load_model(cls, filepath):
        """Load a neural network from a file"""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        with open(filepath, 'r') as f:
            model_data = json.load(f)
        
        # Create network with loaded architecture
        network = cls(
            input_size=model_data['input_size'],
            hidden_size=model_data['hidden_size'],
            output_size=model_data['output_size']
        )
        
        # Set the weights
        weights = np.array(model_data['weights'])
        network.set_weights(weights)
        
        print(f"Model loaded from {filepath}")
        return network
