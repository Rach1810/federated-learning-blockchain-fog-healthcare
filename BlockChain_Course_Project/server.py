import numpy as np
from model import create_model
from collections import defaultdict

class Server:
    def __init__(self, input_dim, num_classes):
        self.global_model = create_model(input_dim, num_classes)
        self.global_model_weights = self.global_model.get_weights()
        
    def aggregate(self, client_weights, client_accuracies):
        # Clip and normalize accuracies
        accs = np.clip(np.array(client_accuracies), 0.7, 0.98)  # Reduce outlier impact
        weights = np.exp(accs) / np.sum(np.exp(accs))
        
        # Adaptive aggregation with smoother momentum
        new_weights = []
        for layer in range(len(client_weights[0])):
            layer_weights = np.zeros_like(client_weights[0][layer])
            
            # Calculate weighted average
            for w, weight in zip(client_weights, weights):
                layer_weights += weight * w[layer]
            
            # Adaptive momentum (more conservative when accuracy drops)
            if hasattr(self, 'global_weights'):
                current_acc = np.mean(client_accuracies)
                momentum = 0.5 if current_acc > 0.8 else 0.7  # Preserve more when accuracy is low
                layer_weights = momentum*self.global_weights[layer] + (1-momentum)*layer_weights
                
            new_weights.append(layer_weights)
        
        self.global_weights = new_weights
        self.global_model.set_weights(new_weights)
        return self.global_model