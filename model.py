"""
Sequential Model for Next Word Prediction
Uses LSTM (Long Short-Term Memory) architecture for text generation
"""

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import numpy as np


class NextWordPredictionModel:
    """
    Sequential LSTM-based model for next word prediction in text generation.
    
    Architecture:
    - Embedding Layer: Maps word indices to dense vectors
    - LSTM Layers: Capture sequential patterns in text
    - Dense Layer: Outputs probability distribution over vocabulary
    """
    
    def __init__(self, vocab_size, embedding_dim=128, lstm_units=256, 
                 num_lstm_layers=2, dropout_rate=0.2):
        """
        Initialize the sequential model.
        
        Args:
            vocab_size: Size of the vocabulary
            embedding_dim: Dimension of word embeddings
            lstm_units: Number of units in each LSTM layer
            num_lstm_layers: Number of LSTM layers
            dropout_rate: Dropout rate for regularization
        """
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.lstm_units = lstm_units
        self.num_lstm_layers = num_lstm_layers
        self.dropout_rate = dropout_rate
        self.model = None
        
    def build_model(self, sequence_length):
        """
        Build the sequential LSTM model architecture.
        
        Args:
            sequence_length: Length of input sequences
        """
        self.model = Sequential([
            # Embedding layer: converts word indices to dense vectors
            Embedding(
                input_dim=self.vocab_size,
                output_dim=self.embedding_dim,
                input_length=sequence_length,
                name='embedding'
            ),
        ])
        
        # Add LSTM layers
        for i in range(self.num_lstm_layers):
            return_sequences = (i < self.num_lstm_layers - 1)
            self.model.add(
                LSTM(
                    units=self.lstm_units,
                    return_sequences=return_sequences,
                    name=f'lstm_{i+1}'
                )
            )
            self.model.add(Dropout(self.dropout_rate, name=f'dropout_{i+1}'))
        
        # Output layer: predicts next word from vocabulary
        self.model.add(
            Dense(
                units=self.vocab_size,
                activation='softmax',
                name='output'
            )
        )
        
        # Compile model
        self.model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return self.model
    
    def get_model_summary(self):
        """Print model architecture summary."""
        if self.model:
            return self.model.summary()
        else:
            print("Model not built yet. Call build_model() first.")
    
    def train(self, X_train, y_train, X_val=None, y_val=None, 
              epochs=50, batch_size=64, verbose=1):
        """
        Train the sequential model.
        
        Args:
            X_train: Training sequences
            y_train: Training labels (next words)
            X_val: Validation sequences (optional)
            y_val: Validation labels (optional)
            epochs: Number of training epochs
            batch_size: Batch size for training
            verbose: Verbosity level
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
        
        # Callbacks for training
        callbacks = [
            EarlyStopping(
                monitor='val_loss' if X_val is not None else 'loss',
                patience=5,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss' if X_val is not None else 'loss',
                factor=0.5,
                patience=3,
                min_lr=0.0001,
                verbose=1
            ),
            ModelCheckpoint(
                'best_model.h5',
                monitor='val_loss' if X_val is not None else 'loss',
                save_best_only=True,
                verbose=1
            )
        ]
        
        # Training
        validation_data = (X_val, y_val) if X_val is not None else None
        
        history = self.model.fit(
            X_train, y_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=verbose
        )
        
        return history
    
    def predict_next_word(self, sequence, top_k=5):
        """
        Predict the next word given a sequence.
        
        Args:
            sequence: Input sequence of word indices
            top_k: Number of top predictions to return
        
        Returns:
            List of (word_index, probability) tuples
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
        
        # Reshape for model input
        sequence = np.array(sequence).reshape(1, -1)
        
        # Get predictions
        predictions = self.model.predict(sequence, verbose=0)[0]
        
        # Get top-k predictions
        top_indices = np.argsort(predictions)[-top_k:][::-1]
        top_predictions = [(idx, float(predictions[idx])) for idx in top_indices]
        
        return top_predictions
    
    def save_model(self, filepath):
        """Save the trained model."""
        if self.model:
            self.model.save(filepath)
            print(f"Model saved to {filepath}")
        else:
            raise ValueError("No model to save.")
    
    def load_model(self, filepath):
        """Load a trained model."""
        self.model = tf.keras.models.load_model(filepath)
        print(f"Model loaded from {filepath}")
