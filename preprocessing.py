"""
Data Preprocessing Utilities for Text Generation
Handles tokenization, sequence creation, and data preparation
"""

import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import os


class TextPreprocessor:
    """
    Preprocesses text data for sequential model training.
    Handles tokenization, sequence generation, and vocabulary management.
    """
    
    def __init__(self, max_vocab_size=10000, sequence_length=50, oov_token="<OOV>"):
        """
        Initialize the preprocessor.
        
        Args:
            max_vocab_size: Maximum vocabulary size
            sequence_length: Length of input sequences
            oov_token: Token for out-of-vocabulary words
        """
        self.max_vocab_size = max_vocab_size
        self.sequence_length = sequence_length
        self.oov_token = oov_token
        self.tokenizer = None
        self.vocab_size = 0
        self.word_to_index = {}
        self.index_to_word = {}
        
    def load_text(self, filepath):
        """
        Load text from a file.
        
        Args:
            filepath: Path to text file
        
        Returns:
            Text content as string
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            text = f.read()
        return text
    
    def preprocess_text(self, text):
        """
        Basic text preprocessing: lowercase and split into words.
        
        Args:
            text: Raw text string
        
        Returns:
            List of words
        """
        # Convert to lowercase and split
        text = text.lower()
        words = text.split()
        return words
    
    def fit_tokenizer(self, texts):
        """
        Fit tokenizer on training texts.
        
        Args:
            texts: List of text strings or single text string
        """
        if isinstance(texts, str):
            texts = [texts]
        
        # Create and fit tokenizer
        self.tokenizer = Tokenizer(
            num_words=self.max_vocab_size,
            oov_token=self.oov_token,
            filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'
        )
        
        self.tokenizer.fit_on_texts(texts)
        
        # Update vocabulary
        self.vocab_size = len(self.tokenizer.word_index) + 1
        self.word_to_index = self.tokenizer.word_index
        self.index_to_word = {v: k for k, v in self.word_to_index.items()}
        
        print(f"Vocabulary size: {self.vocab_size}")
        print(f"Total words in corpus: {self.tokenizer.document_count}")
    
    def text_to_sequences(self, texts):
        """
        Convert texts to sequences of word indices.
        
        Args:
            texts: Text string or list of texts
        
        Returns:
            Numpy array of sequences
        """
        if self.tokenizer is None:
            raise ValueError("Tokenizer not fitted. Call fit_tokenizer() first.")
        
        sequences = self.tokenizer.texts_to_sequences(texts)
        return np.array(sequences)
    
    def create_sequences(self, sequences):
        """
        Create input-output pairs for training.
        For each sequence, create multiple training examples by sliding window.
        
        Args:
            sequences: Array of word index sequences
        
        Returns:
            X: Input sequences
            y: Target words (next words)
        """
        X = []
        y = []
        
        for seq in sequences:
            if len(seq) < self.sequence_length + 1:
                continue
            
            # Create sequences by sliding window
            for i in range(len(seq) - self.sequence_length):
                X.append(seq[i:i + self.sequence_length])
                y.append(seq[i + self.sequence_length])
        
        X = np.array(X)
        y = np.array(y)
        
        return X, y
    
    def prepare_training_data(self, text):
        """
        Complete preprocessing pipeline for training data.
        
        Args:
            text: Raw text string
        
        Returns:
            X: Input sequences
            y: Target words
        """
        # Preprocess text
        words = self.preprocess_text(text)
        text_str = ' '.join(words)
        
        # Fit tokenizer
        self.fit_tokenizer(text_str)
        
        # Convert to sequences
        sequences = self.text_to_sequences([text_str])
        
        # Flatten sequences
        flat_sequences = []
        for seq in sequences:
            flat_sequences.extend(seq)
        
        # Create training sequences
        X, y = self.create_sequences([flat_sequences])
        
        print(f"Created {len(X)} training sequences")
        print(f"Sequence length: {self.sequence_length}")
        
        return X, y
    
    def word_to_index_map(self, word):
        """Convert word to index."""
        if self.tokenizer is None:
            raise ValueError("Tokenizer not fitted.")
        return self.tokenizer.word_index.get(word, 0)
    
    def index_to_word_map(self, index):
        """Convert index to word."""
        if self.tokenizer is None:
            raise ValueError("Tokenizer not fitted.")
        return self.index_to_word.get(index, self.oov_token)
    
    def save_tokenizer(self, filepath):
        """Save tokenizer to file."""
        if self.tokenizer:
            with open(filepath, 'wb') as f:
                pickle.dump(self.tokenizer, f)
            print(f"Tokenizer saved to {filepath}")
        else:
            raise ValueError("No tokenizer to save.")
    
    def load_tokenizer(self, filepath):
        """Load tokenizer from file."""
        with open(filepath, 'rb') as f:
            self.tokenizer = pickle.load(f)
        
        # Update vocabulary
        self.vocab_size = len(self.tokenizer.word_index) + 1
        self.word_to_index = self.tokenizer.word_index
        self.index_to_word = {v: k for k, v in self.word_to_index.items()}
        print(f"Tokenizer loaded from {filepath}")
