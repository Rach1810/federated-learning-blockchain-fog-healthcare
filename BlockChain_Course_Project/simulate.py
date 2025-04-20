from data_loader import load_dataset
from client import Client
from server import Server
from blockchain import Blockchain
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf  # Added for model cloning

NUM_CLIENTS = 5
EPOCHS = 10
ROUNDS = 10

def split_data_among_clients(X, y, num_clients):
    """Stratified split preserving class distribution across clients"""
    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=num_clients)
    client_data = []
    
    for _, test_idx in skf.split(X, y):
        client_data.append((X[test_idx], y[test_idx]))
    return client_data

def main():
    print("🔄 Loading dataset...")
    X_train, y_train, X_test, y_test = load_dataset()
    input_dim = X_train.shape[1]
    num_classes = len(np.unique(y_train))
    
    print(f"📦 Partitioning data among {NUM_CLIENTS} clients...")
    client_data = split_data_among_clients(X_train, y_train, NUM_CLIENTS)
    
    server = Server(input_dim, num_classes)
    blockchain = Blockchain()
    
    global_accuracies = []
    best_accuracy = 0
    best_model = None
    
    for round_num in range(1, ROUNDS + 1):
        print(f"\n🚀 Round {round_num} - Training on clients...")
        client_weights = []
        client_accuracies = []

        for client_id, (x, y) in enumerate(client_data):
            client = Client(
                client_id=f"Client-{client_id+1}",
                x_data=x,
                y_data=y,
                input_dim=input_dim,
                num_classes=num_classes
            )
            weights, acc = client.train(epochs=EPOCHS)
            client_weights.append(weights)
            client_accuracies.append(acc)
            blockchain.add_block(client.client_id, acc)
            print(f"  ✅ {client.client_id} trained with val accuracy: {acc:.4f}")
        
        print("🔁 Aggregating global model on server...")
        global_model = server.aggregate(client_weights, client_accuracies)
        
        print("🧪 Evaluating global model...")
        loss, accuracy = global_model.evaluate(X_test, y_test, verbose=0)
        global_accuracies.append(accuracy)
        
        # Track best model
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = tf.keras.models.clone_model(global_model)
            best_model.set_weights(global_model.get_weights())
        
        print(f"🌍 Global Model Accuracy: {accuracy:.4f} | Best: {best_accuracy:.4f}")
    
    # Plot accuracy progression
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, ROUNDS+1), global_accuracies, marker='o')
    plt.title("Global Model Accuracy Progression")
    plt.xlabel("Communication Round")
    plt.ylabel("Test Accuracy")
    plt.xticks(range(1, ROUNDS+1))
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()
    
    # Evaluate best model
    if best_model is not None:
        print("\n🏆 Best Model Evaluation:")
        test_loss, test_acc = best_model.evaluate(X_test, y_test, verbose=0)
        print(f"Final Test Accuracy: {test_acc:.4f}")
    else:
        print("\n⚠️ No best model was saved during training")
    
    print("\n🔗 Blockchain Log:")
    blockchain.print_chain()

if __name__ == "__main__":
    main()