"""
Result Visualization Script
Generates comprehensive diagrams for research paper presentation
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import pickle
from model import NextWordPredictionModel
from preprocessing import TextPreprocessor


def plot_training_history_comprehensive(history, save_dir='models'):
    """
    Create comprehensive training history plots with multiple metrics.
    
    Args:
        history: Training history object from Keras
        save_dir: Directory to save plots
    """
    fig = plt.figure(figsize=(16, 10))
    
    # 1. Loss curves
    ax1 = plt.subplot(2, 3, 1)
    ax1.plot(history.history['loss'], label='Training Loss', linewidth=2, color='#2E86AB')
    if 'val_loss' in history.history:
        ax1.plot(history.history['val_loss'], label='Validation Loss', linewidth=2, color='#A23B72')
    ax1.set_title('Model Loss Over Epochs', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_facecolor('#F5F5F5')
    
    # 2. Accuracy curves
    ax2 = plt.subplot(2, 3, 2)
    ax2.plot(history.history['accuracy'], label='Training Accuracy', linewidth=2, color='#06A77D')
    if 'val_accuracy' in history.history:
        ax2.plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2, color='#F18F01')
    ax2.set_title('Model Accuracy Over Epochs', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_facecolor('#F5F5F5')
    
    # 3. Loss comparison (bar chart for final values)
    ax3 = plt.subplot(2, 3, 3)
    metrics = []
    values = []
    colors = []
    
    metrics.append('Final Train Loss')
    values.append(history.history['loss'][-1])
    colors.append('#2E86AB')
    
    if 'val_loss' in history.history:
        metrics.append('Final Val Loss')
        values.append(history.history['val_loss'][-1])
        colors.append('#A23B72')
    
    bars = ax3.bar(metrics, values, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    ax3.set_title('Final Loss Comparison', fontsize=14, fontweight='bold')
    ax3.set_ylabel('Loss', fontsize=12)
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.set_facecolor('#F5F5F5')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # 4. Accuracy comparison
    ax4 = plt.subplot(2, 3, 4)
    metrics = []
    values = []
    colors = []
    
    metrics.append('Final Train Acc')
    values.append(history.history['accuracy'][-1])
    colors.append('#06A77D')
    
    if 'val_accuracy' in history.history:
        metrics.append('Final Val Acc')
        values.append(history.history['val_accuracy'][-1])
        colors.append('#F18F01')
    
    bars = ax4.bar(metrics, values, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    ax4.set_title('Final Accuracy Comparison', fontsize=14, fontweight='bold')
    ax4.set_ylabel('Accuracy', fontsize=12)
    ax4.set_ylim([0, 1])
    ax4.grid(True, alpha=0.3, axis='y')
    ax4.set_facecolor('#F5F5F5')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # 5. Overfitting analysis (gap between train and val)
    ax5 = plt.subplot(2, 3, 5)
    if 'val_loss' in history.history and 'val_accuracy' in history.history:
        epochs = range(1, len(history.history['loss']) + 1)
        loss_gap = np.array(history.history['val_loss']) - np.array(history.history['loss'])
        acc_gap = np.array(history.history['accuracy']) - np.array(history.history['val_accuracy'])
        
        ax5_twin = ax5.twinx()
        line1 = ax5.plot(epochs, loss_gap, label='Loss Gap (Val-Train)', 
                        color='#A23B72', linewidth=2, marker='o', markersize=4)
        line2 = ax5_twin.plot(epochs, acc_gap, label='Accuracy Gap (Train-Val)', 
                             color='#F18F01', linewidth=2, marker='s', markersize=4)
        
        ax5.set_xlabel('Epoch', fontsize=12)
        ax5.set_ylabel('Loss Gap', fontsize=12, color='#A23B72')
        ax5_twin.set_ylabel('Accuracy Gap', fontsize=12, color='#F18F01')
        ax5.set_title('Overfitting Analysis', fontsize=14, fontweight='bold')
        ax5.grid(True, alpha=0.3)
        ax5.set_facecolor('#F5F5F5')
        
        # Combine legends
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax5.legend(lines, labels, loc='upper left', fontsize=9)
    
    # 6. Training progress (combined view)
    ax6 = plt.subplot(2, 3, 6)
    epochs = range(1, len(history.history['loss']) + 1)
    
    ax6_twin = ax6.twinx()
    line1 = ax6.plot(epochs, history.history['loss'], label='Loss', 
                    color='#2E86AB', linewidth=2)
    if 'val_loss' in history.history:
        ax6.plot(epochs, history.history['val_loss'], label='Val Loss', 
                color='#A23B72', linewidth=2, linestyle='--')
    line2 = ax6_twin.plot(epochs, history.history['accuracy'], label='Accuracy', 
                         color='#06A77D', linewidth=2)
    if 'val_accuracy' in history.history:
        ax6_twin.plot(epochs, history.history['val_accuracy'], label='Val Accuracy', 
                     color='#F18F01', linewidth=2, linestyle='--')
    
    ax6.set_xlabel('Epoch', fontsize=12)
    ax6.set_ylabel('Loss', fontsize=12, color='#2E86AB')
    ax6_twin.set_ylabel('Accuracy', fontsize=12, color='#06A77D')
    ax6.set_title('Training Progress Overview', fontsize=14, fontweight='bold')
    ax6.grid(True, alpha=0.3)
    ax6.set_facecolor('#F5F5F5')
    
    # Combine legends
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax6.legend(lines, labels, loc='center right', fontsize=9)
    
    plt.suptitle('Comprehensive Training Results Analysis', 
                fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    
    plot_path = os.path.join(save_dir, 'comprehensive_training_results.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Comprehensive training results saved to: {plot_path}")
    plt.close()


def plot_model_architecture(save_dir='models'):
    """
    Create a visual diagram of the model architecture.
    
    Args:
        save_dir: Directory to save the plot
    """
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)
    ax.axis('off')
    
    # Colors
    embed_color = '#4A90E2'
    lstm_color = '#50C878'
    dense_color = '#FF6B6B'
    text_color = '#2C3E50'
    
    # Title
    ax.text(5, 11.5, 'Sequential LSTM Model Architecture', 
           fontsize=18, fontweight='bold', ha='center', color=text_color)
    
    # Input layer
    input_box = FancyBboxPatch((3.5, 9.5), 3, 0.8, 
                               boxstyle="round,pad=0.1", 
                               facecolor='#E8E8E8', edgecolor='black', linewidth=2)
    ax.add_patch(input_box)
    ax.text(5, 9.9, 'Input Sequences\n(Sequence Length: 50)', 
           fontsize=11, ha='center', va='center', fontweight='bold')
    
    # Embedding layer
    embed_box = FancyBboxPatch((3.5, 8), 3, 0.8, 
                              boxstyle="round,pad=0.1", 
                              facecolor=embed_color, edgecolor='black', linewidth=2, alpha=0.8)
    ax.add_patch(embed_box)
    ax.text(5, 8.4, 'Embedding Layer\n(128 dimensions)', 
           fontsize=11, ha='center', va='center', fontweight='bold', color='white')
    
    # Arrow 1
    arrow1 = FancyArrowPatch((5, 8), (5, 7.5), 
                            arrowstyle='->', mutation_scale=20, 
                            linewidth=2, color='black')
    ax.add_patch(arrow1)
    
    # LSTM Layer 1
    lstm1_box = FancyBboxPatch((3.5, 6.2), 3, 0.8, 
                              boxstyle="round,pad=0.1", 
                              facecolor=lstm_color, edgecolor='black', linewidth=2, alpha=0.8)
    ax.add_patch(lstm1_box)
    ax.text(5, 6.6, 'LSTM Layer 1\n(256 units)', 
           fontsize=11, ha='center', va='center', fontweight='bold', color='white')
    
    # Dropout 1
    dropout1_box = FancyBboxPatch((3.5, 5.2), 3, 0.5, 
                                 boxstyle="round,pad=0.1", 
                                 facecolor='#FFA500', edgecolor='black', linewidth=1.5, alpha=0.7)
    ax.add_patch(dropout1_box)
    ax.text(5, 5.45, 'Dropout (0.2)', 
           fontsize=10, ha='center', va='center', fontweight='bold')
    
    # Arrow 2
    arrow2 = FancyArrowPatch((5, 5.2), (5, 4.7), 
                            arrowstyle='->', mutation_scale=20, 
                            linewidth=2, color='black')
    ax.add_patch(arrow2)
    
    # LSTM Layer 2
    lstm2_box = FancyBboxPatch((3.5, 3.4), 3, 0.8, 
                              boxstyle="round,pad=0.1", 
                              facecolor=lstm_color, edgecolor='black', linewidth=2, alpha=0.8)
    ax.add_patch(lstm2_box)
    ax.text(5, 3.8, 'LSTM Layer 2\n(256 units)', 
           fontsize=11, ha='center', va='center', fontweight='bold', color='white')
    
    # Dropout 2
    dropout2_box = FancyBboxPatch((3.5, 2.4), 3, 0.5, 
                                 boxstyle="round,pad=0.1", 
                                 facecolor='#FFA500', edgecolor='black', linewidth=1.5, alpha=0.7)
    ax.add_patch(dropout2_box)
    ax.text(5, 2.65, 'Dropout (0.2)', 
           fontsize=10, ha='center', va='center', fontweight='bold')
    
    # Arrow 3
    arrow3 = FancyArrowPatch((5, 2.4), (5, 1.9), 
                            arrowstyle='->', mutation_scale=20, 
                            linewidth=2, color='black')
    ax.add_patch(arrow3)
    
    # Output layer
    output_box = FancyBboxPatch((3.5, 0.5), 3, 0.8, 
                               boxstyle="round,pad=0.1", 
                               facecolor=dense_color, edgecolor='black', linewidth=2, alpha=0.8)
    ax.add_patch(output_box)
    ax.text(5, 0.9, 'Dense Output Layer\n(Softmax, Vocab Size)', 
           fontsize=11, ha='center', va='center', fontweight='bold', color='white')
    
    # Side annotations
    ax.text(1, 6, 'Model Parameters:\n\n• Embedding Dim: 128\n• LSTM Units: 256\n• LSTM Layers: 2\n• Dropout: 0.2\n• Optimizer: Adam\n• Learning Rate: 0.001', 
           fontsize=10, va='center', bbox=dict(boxstyle='round', facecolor='#F0F0F0', alpha=0.8))
    
    ax.text(8.5, 6, 'Key Features:\n\n• Sequential Processing\n• Long-term Memory\n• Context Preservation\n• Probability Distribution\n• Next Word Prediction', 
           fontsize=10, va='center', bbox=dict(boxstyle='round', facecolor='#F0F0F0', alpha=0.8))
    
    plt.tight_layout()
    plot_path = os.path.join(save_dir, 'model_architecture_diagram.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Model architecture diagram saved to: {plot_path}")
    plt.close()


def plot_prediction_examples(model_path='models/next_word_model.h5',
                             tokenizer_path='models/tokenizer.pkl',
                             seed_texts=None,
                             save_dir='models'):
    """
    Visualize prediction examples with confidence scores.
    
    Args:
        model_path: Path to trained model
        tokenizer_path: Path to tokenizer
        seed_texts: List of seed texts to generate from
        save_dir: Directory to save plots
    """
    if seed_texts is None:
        seed_texts = [
            "machine learning is",
            "the future of",
            "artificial intelligence",
            "deep learning models"
        ]
    
    # Load model and tokenizer
    preprocessor = TextPreprocessor(sequence_length=50)
    preprocessor.load_tokenizer(tokenizer_path)
    
    model = NextWordPredictionModel(vocab_size=preprocessor.vocab_size)
    model.load_model(model_path)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    for idx, seed_text in enumerate(seed_texts):
        ax = axes[idx]
        
        # Preprocess seed text
        seed_words = preprocessor.preprocess_text(seed_text)
        seed_sequence = []
        for word in seed_words:
            idx_word = preprocessor.word_to_index_map(word)
            if idx_word > 0:
                seed_sequence.append(idx_word)
        
        if len(seed_sequence) < 50:
            seed_sequence = [0] * (50 - len(seed_sequence)) + seed_sequence
        else:
            seed_sequence = seed_sequence[-50:]
        
        # Get predictions
        predictions = model.predict_next_word(seed_sequence, top_k=10)
        
        # Extract top predictions
        words = [preprocessor.index_to_word_map(p[0]) for p in predictions]
        probs = [p[1] for p in predictions]
        
        # Create bar chart
        bars = ax.barh(range(len(words)), probs, color=plt.cm.viridis(np.linspace(0, 1, len(words))))
        ax.set_yticks(range(len(words)))
        ax.set_yticklabels(words, fontsize=10)
        ax.set_xlabel('Probability', fontsize=11, fontweight='bold')
        ax.set_title(f'Top 10 Predictions for: "{seed_text}"', 
                    fontsize=12, fontweight='bold', pad=10)
        ax.grid(True, alpha=0.3, axis='x')
        ax.set_facecolor('#F5F5F5')
        
        # Add probability labels
        for i, (bar, prob) in enumerate(zip(bars, probs)):
            ax.text(prob + 0.01, i, f'{prob:.4f}', 
                   va='center', fontsize=9, fontweight='bold')
    
    plt.suptitle('Next Word Prediction Examples with Confidence Scores', 
                fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    
    plot_path = os.path.join(save_dir, 'prediction_examples.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Prediction examples saved to: {plot_path}")
    plt.close()


def generate_all_diagrams(history_path=None, model_path='models/next_word_model.h5',
                         tokenizer_path='models/tokenizer.pkl', save_dir='models'):
    """
    Generate all result diagrams for research paper.
    
    Args:
        history_path: Path to saved training history (optional)
        model_path: Path to trained model
        tokenizer_path: Path to tokenizer
        save_dir: Directory to save all diagrams
    """
    print("=" * 60)
    print("Generating Result Diagrams for Research Paper")
    print("=" * 60)
    
    # Load history if provided
    if history_path and os.path.exists(history_path):
        with open(history_path, 'rb') as f:
            history = pickle.load(f)
    else:
        print("Note: Training history not found. Some diagrams will be skipped.")
        print("Run training first to generate complete diagrams.")
        history = None
    
    # Generate diagrams
    if history:
        print("\n[1/3] Generating comprehensive training results...")
        plot_training_history_comprehensive(history, save_dir)
    
    print("\n[2/3] Generating model architecture diagram...")
    plot_model_architecture(save_dir)
    
    if os.path.exists(model_path) and os.path.exists(tokenizer_path):
        print("\n[3/3] Generating prediction examples...")
        plot_prediction_examples(model_path, tokenizer_path, save_dir=save_dir)
    else:
        print("\n[3/3] Skipping prediction examples (model not found)")
    
    print("\n" + "=" * 60)
    print("All diagrams generated successfully!")
    print(f"Diagrams saved in: {save_dir}")
    print("=" * 60)


if __name__ == "__main__":
    # Generate all diagrams
    generate_all_diagrams()
