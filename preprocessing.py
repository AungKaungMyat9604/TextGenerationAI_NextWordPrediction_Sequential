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
            
            # For large sequences, use chunked approach
            if len(seq) > 1000000:
                X_chunk, y_chunk = self.create_sequences_chunked(seq)
                X.extend(X_chunk)
                y.extend(y_chunk)
            else:
                # Create sequences by sliding window for smaller sequences
                for i in range(len(seq) - self.sequence_length):
                    X.append(seq[i:i + self.sequence_length])
                    y.append(seq[i + self.sequence_length])
        
        X = np.array(X, dtype=np.int32)
        y = np.array(y, dtype=np.int32)
        
        return X, y
    
    def prepare_training_data(self, text, max_words=None, max_chars=None):
        """
        Complete preprocessing pipeline for training data.
        
        Args:
            text: Raw text string
            max_words: Maximum number of words to process (None for all)
            max_chars: Maximum number of characters to process (None for all)
        
        Returns:
            X: Input sequences
            y: Target words
        """
        # Limit text size if specified
        if max_chars is not None and len(text) > max_chars:
            print(f"Limiting text to {max_chars:,} characters (from {len(text):,})")
            text = text[:max_chars]
        
        # Preprocess text
        words = self.preprocess_text(text)
        
        # Limit number of words if specified
        if max_words is not None and len(words) > max_words:
            print(f"Limiting text to {max_words:,} words (from {len(words):,})")
            words = words[:max_words]
        
        text_str = ' '.join(words)
        
        # Fit tokenizer
        print("Fitting tokenizer...")
        self.fit_tokenizer(text_str)
        
        # Convert to sequences
        print("Converting text to sequences...")
        sequences = self.text_to_sequences([text_str])
        
        # Flatten sequences
        flat_sequences = []
        for seq in sequences:
            flat_sequences.extend(seq)
        
        # Create training sequences using chunked approach for large datasets
        print("Creating training sequences (this may take a while for large datasets)...")
        X, y = self.create_sequences_chunked(flat_sequences)
        
        print(f"Created {len(X):,} training sequences")
        print(f"Sequence length: {self.sequence_length}")
        
        return X, y
    
    def create_sequences_chunked(self, flat_sequence, chunk_size=1000000):
        """
        Create input-output pairs using chunked processing to save memory.
        
        Args:
            flat_sequence: Flat list of word indices
            chunk_size: Process sequences in chunks of this size
        
        Returns:
            X: Input sequences
            y: Target words
        """
        X = []
        y = []
        
        total_length = len(flat_sequence)
        if total_length < self.sequence_length + 1:
            return np.array([]), np.array([])
        
        # Process in chunks to avoid memory issues
        num_chunks = (total_length // chunk_size) + 1
        
        for chunk_idx in range(num_chunks):
            start_idx = chunk_idx * chunk_size
            end_idx = min(start_idx + chunk_size + self.sequence_length, total_length)
            
            if end_idx - start_idx < self.sequence_length + 1:
                break
            
            chunk = flat_sequence[start_idx:end_idx]
            
            # Create sequences for this chunk
            for i in range(len(chunk) - self.sequence_length):
                X.append(chunk[i:i + self.sequence_length])
                y.append(chunk[i + self.sequence_length])
            
            if (chunk_idx + 1) % 10 == 0:
                print(f"  Processed chunk {chunk_idx + 1}/{num_chunks} ({len(X):,} sequences so far)...")
        
        X = np.array(X, dtype=np.int32)
        y = np.array(y, dtype=np.int32)
        
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
