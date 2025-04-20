import tensorflow as tf  
from model import create_model
import numpy as np
from sklearn.model_selection import train_test_split

class Client:
    def __init__(self, client_id, x_data, y_data, input_dim, num_classes):
        self.client_id = client_id
        self.input_dim = input_dim
        self.num_classes = num_classes
        
        # Split local data into train/validation
        self.x_train, self.x_val, self.y_train, self.y_val = train_test_split(
            x_data, y_data, test_size=0.2, random_state=42)
        
        self.model = create_model(input_dim, num_classes)
        self.best_weights = None

    def train(self, epochs=10, batch_size=32):
        # Dynamic learning rate scheduling
        callbacks = [
            tf.keras.callbacks.EarlyStopping(patience=3, monitor='val_loss'),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss', 
                factor=0.5, 
                patience=2,
                min_lr=1e-5)
        ]
        
        # Gradient clipping during training
        opt = tf.keras.optimizers.Adam(
            learning_rate=0.0003,
            clipnorm=1.0)
        self.model.compile(
            optimizer=opt,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'])
        
        history = self.model.fit(
            self.x_train, self.y_train,
            validation_data=(self.x_val, self.y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=0
        )
        return self.model.get_weights(), history.history['val_accuracy'][-1]