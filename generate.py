"""
Text Generation Script
Generates text using the trained sequential model
"""

import os
import numpy as np
from model import NextWordPredictionModel
from preprocessing import TextPreprocessor


def generate_text(seed_text, model_path='models/next_word_model.h5',
                  tokenizer_path='models/tokenizer.pkl',
                  sequence_length=50,
                  num_words=50,
                  temperature=1.0,
                  top_k=5):
    """
    Generate text using the trained model.
    
    Args:
        seed_text: Initial text to start generation
        model_path: Path to trained model
        tokenizer_path: Path to tokenizer
        sequence_length: Length of input sequences (must match training)
        num_words: Number of words to generate
        temperature: Sampling temperature (higher = more random)
        top_k: Consider top k predictions
    
    Returns:
        Generated text string
    """
    
    print("=" * 60)
    print("Text Generation")
    print("=" * 60)
    
    # Load tokenizer
    print(f"\nLoading tokenizer from {tokenizer_path}...")
    preprocessor = TextPreprocessor(sequence_length=sequence_length)
    preprocessor.load_tokenizer(tokenizer_path)
    
    # Load model
    print(f"Loading model from {model_path}...")
    model = NextWordPredictionModel(vocab_size=preprocessor.vocab_size)
    model.load_model(model_path)
    
    # Preprocess seed text
    print(f"\nSeed text: '{seed_text}'")
    seed_words = preprocessor.preprocess_text(seed_text)
    
    # Convert seed text to sequence
    seed_sequence = []
    for word in seed_words:
        idx = preprocessor.word_to_index_map(word)
        if idx > 0:  # Skip OOV tokens
            seed_sequence.append(idx)
    
    # Pad or truncate to sequence_length
    if len(seed_sequence) < sequence_length:
        # Pad with zeros (or repeat)
        seed_sequence = [0] * (sequence_length - len(seed_sequence)) + seed_sequence
    else:
        seed_sequence = seed_sequence[-sequence_length:]
    
    generated_text = seed_text
    current_sequence = seed_sequence.copy()
    
    print(f"\nGenerating {num_words} words...")
    print("-" * 60)
    
    for i in range(num_words):
        # Get predictions
        predictions = model.predict_next_word(current_sequence, top_k=top_k)
        
        # Apply temperature sampling
        if temperature > 0:
            # Extract probabilities
            indices = [p[0] for p in predictions]
            probs = np.array([p[1] for p in predictions])
            
            # Apply temperature
            probs = np.power(probs, 1.0 / temperature)
            probs = probs / np.sum(probs)
            
            # Sample from distribution
            next_idx = np.random.choice(indices, p=probs)
        else:
            # Greedy selection
            next_idx = predictions[0][0]
        
        # Convert to word
        next_word = preprocessor.index_to_word_map(next_idx)
        
        # Update sequence
        current_sequence.append(next_idx)
        current_sequence = current_sequence[-sequence_length:]
        
        # Append to generated text
        generated_text += " " + next_word
        
        # Print progress
        if (i + 1) % 10 == 0:
            print(f"Generated {i + 1}/{num_words} words...")
    
    print("-" * 60)
    print("\nGenerated Text:")
    print("=" * 60)
    print(generated_text)
    print("=" * 60)
    
    return generated_text


def interactive_generation(model_path='models/next_word_model.h5',
                          tokenizer_path='models/tokenizer.pkl',
                          sequence_length=50):
    """
    Interactive text generation mode.
    """
    print("=" * 60)
    print("Interactive Text Generation")
    print("Type 'quit' to exit")
    print("=" * 60)
    
    # Load model and tokenizer
    preprocessor = TextPreprocessor(sequence_length=sequence_length)
    preprocessor.load_tokenizer(tokenizer_path)
    
    model = NextWordPredictionModel(vocab_size=preprocessor.vocab_size)
    model.load_model(model_path)
    
    while True:
        seed_text = input("\nEnter seed text (or 'quit' to exit): ")
        if seed_text.lower() == 'quit':
            break
        
        try:
            num_words = int(input("Number of words to generate (default 20): ") or "20")
            generated = generate_text(
                seed_text,
                model_path=model_path,
                tokenizer_path=tokenizer_path,
                sequence_length=sequence_length,
                num_words=num_words
            )
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # Command line mode
        seed_text = sys.argv[1]
        num_words = int(sys.argv[2]) if len(sys.argv) > 2 else 50
        
        generate_text(
            seed_text=seed_text,
            num_words=num_words
        )
    else:
        # Interactive mode
        interactive_generation()
