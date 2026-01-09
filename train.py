"""
Training Script for Next Word Prediction Model
"""

import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from model import NextWordPredictionModel
from preprocessing import TextPreprocessor


def train_model(text_file='data/sample_text.txt', 
                sequence_length=50,
                embedding_dim=128,
                lstm_units=256,
                num_lstm_layers=2,
                dropout_rate=0.2,
                epochs=50,
                batch_size=64,
                validation_split=0.2,
                save_dir='models',
                use_bhc_dataset=True,
                bhc_data_dir='data/bhc_mimic_iv',
                bhc_text_column='input'):
    """
    Train the sequential model for next word prediction.
    
    Args:
        text_file: Path to training text file
        sequence_length: Length of input sequences
        embedding_dim: Dimension of word embeddings
        lstm_units: Number of units in LSTM layers
        num_lstm_layers: Number of LSTM layers
        dropout_rate: Dropout rate
        epochs: Number of training epochs
        batch_size: Batch size
        validation_split: Fraction of data for validation
        save_dir: Directory to save model and tokenizer
        use_bhc_dataset: If True, load text from BHC MIMIC-IV Kaggle dataset
        bhc_data_dir: Directory where the BHC dataset CSV files are stored
        bhc_text_column: Column name in the BHC dataset to use as text
    """
    
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs('data', exist_ok=True)
    
    print("=" * 60)
    print("Next Word Prediction Model Training")
    print("=" * 60)
    
    # Initialize preprocessor
    print("\n[1/5] Initializing preprocessor...")
    preprocessor = TextPreprocessor(
        max_vocab_size=10000,
        sequence_length=sequence_length
    )
    
    # Load and preprocess data
    if use_bhc_dataset:
        print(f"\n[2/5] Loading and preprocessing BHC MIMIC-IV data from {bhc_data_dir} (column '{bhc_text_column}')...")
        text = load_bhc_corpus(bhc_data_dir=bhc_data_dir, text_column=bhc_text_column)
    else:
        print(f"\n[2/5] Loading and preprocessing data from {text_file}...")
        if not os.path.exists(text_file):
            print(f"Warning: {text_file} not found. Creating sample text file...")
            create_sample_data(text_file)
        
        text = preprocessor.load_text(text_file)
    print(f"Text length: {len(text)} characters")
    print(f"Text length: {len(text.split())} words")
    
    # Prepare training data
    print("\n[3/5] Preparing training sequences...")
    X, y = preprocessor.prepare_training_data(text)
    
    # Split into train and validation sets
    split_idx = int(len(X) * (1 - validation_split))
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    
    # Build model
    print("\n[4/5] Building model...")
    model = NextWordPredictionModel(
        vocab_size=preprocessor.vocab_size,
        embedding_dim=embedding_dim,
        lstm_units=lstm_units,
        num_lstm_layers=num_lstm_layers,
        dropout_rate=dropout_rate
    )
    
    model.build_model(sequence_length)
    print("\nModel Architecture:")
    model.get_model_summary()
    
    # Train model
    print("\n[5/5] Training model...")
    history = model.train(
        X_train, y_train,
        X_val=X_val, y_val=y_val,
        epochs=epochs,
        batch_size=batch_size,
        verbose=1
    )
    
    # Save model and tokenizer
    print("\nSaving model and tokenizer...")
    model_path = os.path.join(save_dir, 'next_word_model.h5')
    tokenizer_path = os.path.join(save_dir, 'tokenizer.pkl')
    
    model.save_model(model_path)
    preprocessor.save_tokenizer(tokenizer_path)
    
    # Plot training history
    plot_training_history(history, save_dir)
    
    # Save training history for later visualization
    import pickle
    history_path = os.path.join(save_dir, 'training_history.pkl')
    with open(history_path, 'wb') as f:
        pickle.dump(history, f)
    print(f"Training history saved to: {history_path}")
    
    print("\n" + "=" * 60)
    print("Training completed successfully!")
    print(f"Model saved to: {model_path}")
    print(f"Tokenizer saved to: {tokenizer_path}")
    print(f"Training history saved to: {history_path}")
    print("\nTo generate comprehensive result diagrams, run:")
    print("  python visualize_results.py")
    print("=" * 60)
    
    return model, preprocessor, history


def plot_training_history(history, save_dir):
    """Plot training history (loss and accuracy)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot loss
    ax1.plot(history.history['loss'], label='Training Loss')
    if 'val_loss' in history.history:
        ax1.plot(history.history['val_loss'], label='Validation Loss')
    ax1.set_title('Model Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Plot accuracy
    ax2.plot(history.history['accuracy'], label='Training Accuracy')
    if 'val_accuracy' in history.history:
        ax2.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax2.set_title('Model Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plot_path = os.path.join(save_dir, 'training_history.png')
    plt.savefig(plot_path)
    print(f"Training history plot saved to: {plot_path}")
    plt.close()


def create_sample_data(filepath):
    """Create a sample text file for training if none exists."""
    sample_text = """
    The quick brown fox jumps over the lazy dog. 
    Machine learning is a subset of artificial intelligence that focuses on 
    algorithms and statistical models that enable computer systems to improve 
    their performance on a specific task through experience. Deep learning, 
    a subset of machine learning, uses neural networks with multiple layers 
    to learn representations of data. Natural language processing is a field 
    that combines computational linguistics with machine learning to enable 
    computers to understand and generate human language. Text generation is 
    one of the most exciting applications of natural language processing, 
    allowing computers to create coherent and contextually relevant text. 
    Sequential models like LSTM and GRU are particularly well-suited for 
    text generation tasks because they can capture long-term dependencies 
    in sequential data. These models process text one word at a time, 
    maintaining a hidden state that encodes information about previous words 
    in the sequence. This allows them to generate text that is not only 
    grammatically correct but also contextually appropriate. The training 
    process involves feeding the model large amounts of text data and 
    adjusting the model parameters to minimize the prediction error. 
    Through this process, the model learns the statistical patterns and 
    relationships between words in the training corpus. Once trained, 
    the model can generate new text by predicting the next word given 
    a sequence of previous words. This process can be repeated iteratively 
    to generate longer sequences of text. The quality of generated text 
    depends on various factors including the size and quality of the 
    training data, the model architecture, and the training procedure.
    """
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(sample_text)
    print(f"Sample text file created at {filepath}")


def load_bhc_corpus(bhc_data_dir='data/bhc_mimic_iv', text_column='input'):
    """
    Load and concatenate text from the BHC MIMIC-IV Kaggle dataset.
    
    Args:
        bhc_data_dir: Directory containing the downloaded Kaggle dataset files.
        text_column: Name of the column to use as text (e.g., 'summary').
    
    Returns:
        A single large string containing all texts concatenated.
    """
    data_path = Path(bhc_data_dir)
    if not data_path.exists():
        raise FileNotFoundError(
            f"BHC data directory '{bhc_data_dir}' not found. "
            "Make sure you downloaded the Kaggle dataset into this folder."
        )
    
    csv_files = list(data_path.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files found in '{bhc_data_dir}'. "
            "After downloading from Kaggle, you should have at least one CSV file here."
        )
    
    # For now, use the first CSV file. Adjust if your dataset has multiple parts.
    file_path = csv_files[0]
    print(f"Reading BHC dataset from: {file_path}")
    
    df = pd.read_csv(file_path)
    if text_column not in df.columns:
        raise ValueError(
            f"Column '{text_column}' not found in BHC dataset. "
            f"Available columns: {df.columns.tolist()}"
        )
    
    texts = df[text_column].astype(str).tolist()
    corpus = "\n\n".join(texts)
    return corpus


if __name__ == "__main__":
    # Training configuration
    train_model(
        text_file='data/sample_text.txt',
        sequence_length=50,
        embedding_dim=128,
        lstm_units=256,
        num_lstm_layers=2,
        dropout_rate=0.2,
        epochs=50,
        batch_size=64,
        validation_split=0.2,
        save_dir='models',
        # Set this to True to train directly on the Kaggle BHC dataset
        use_bhc_dataset=True,
        bhc_data_dir='data/bhc_mimic_iv',
        bhc_text_column='input'
    )
