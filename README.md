# Text Generation AI - Next Word Prediction with Sequential Model

A deep learning project for text generation using sequential LSTM (Long Short-Term Memory) models for next word prediction.

## Project Overview

This project implements a sequential neural network model for predicting the next word in a sequence, enabling text generation capabilities. The model uses LSTM layers to capture long-term dependencies in text data, making it suitable for generating coherent and contextually relevant text.

## Features

- **Sequential LSTM Architecture**: Multi-layer LSTM network for capturing sequential patterns
- **Word Embeddings**: Dense vector representations of words
- **Text Preprocessing**: Tokenization and sequence generation utilities
- **Model Training**: Complete training pipeline with validation and callbacks
- **Text Generation**: Generate text from seed phrases
- **Interactive Mode**: Command-line interface for text generation

## Project Structure

```
TextGenerationAI_NextWordPrediction_Sequential/
│
├── model.py              # Sequential LSTM model architecture
├── preprocessing.py      # Text preprocessing and tokenization
├── train.py             # Training script
├── generate.py          # Text generation script
├── requirements.txt     # Python dependencies
├── README.md            # Project documentation
├── research_paper_explanation.txt  # Research paper explanation
│
├── data/                # Training data directory
│   └── sample_text.txt  # Sample training text
│
└── models/              # Saved models directory
    ├── next_word_model.h5
    ├── tokenizer.pkl
    └── training_history.png
```

## Installation

1. Clone or download this repository

2. Create a virtual environment (recommended):
```bash
python3 -m venv venv
```

3. Activate the virtual environment:
   - On macOS/Linux:
     ```bash
     source venv/bin/activate
     ```
   - On Windows:
     ```bash
     venv\Scripts\activate
     ```

4. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### 1. Prepare Training Data

Place your training text file in the `data/` directory, or use the sample text that will be created automatically.

### 2. Train the Model

Train the sequential model:
```bash
python train.py
```

You can modify training parameters in `train.py`:
- `sequence_length`: Length of input sequences (default: 50)
- `embedding_dim`: Word embedding dimension (default: 128)
- `lstm_units`: Number of LSTM units (default: 256)
- `num_lstm_layers`: Number of LSTM layers (default: 2)
- `epochs`: Training epochs (default: 50)
- `batch_size`: Batch size (default: 64)

### 3. Generate Text

#### Command Line Mode:
```bash
python generate.py "your seed text here" 50
```

#### Interactive Mode:
```bash
python generate.py
```

Then enter your seed text and number of words to generate when prompted.

## Model Architecture

The sequential model consists of:

1. **Embedding Layer**: Maps word indices to dense vectors (128 dimensions)
2. **LSTM Layers**: Two LSTM layers (256 units each) with dropout regularization
3. **Dense Output Layer**: Softmax layer predicting probability distribution over vocabulary

### Key Components:

- **Embedding Dimension**: 128
- **LSTM Units**: 256 per layer
- **LSTM Layers**: 2
- **Dropout Rate**: 0.2
- **Optimizer**: Adam (learning rate: 0.001)
- **Loss Function**: Sparse Categorical Crossentropy

## Training Process

The training process includes:

- **Data Preprocessing**: Tokenization and sequence creation
- **Train/Validation Split**: 80/20 split by default
- **Callbacks**:
  - Early stopping to prevent overfitting
  - Learning rate reduction on plateau
  - Model checkpointing for best model
- **Metrics**: Loss and accuracy tracking

## Results

After training, you'll find:
- `models/next_word_model.h5`: Trained model weights
- `models/tokenizer.pkl`: Fitted tokenizer
- `models/training_history.pkl`: Training history data
- `models/training_history.png`: Basic training curves

### Generate Comprehensive Result Diagrams

For research paper presentation, generate all result diagrams:

```bash
python visualize_results.py
```

This will create:
- `comprehensive_training_results.png`: Multi-panel training analysis
- `model_architecture_diagram.png`: Visual architecture representation
- `prediction_examples.png`: Sample predictions with confidence scores

All diagrams are saved in high resolution (300 DPI) suitable for research papers.

## Technical Details

### Sequential Model Advantages:
- **Memory**: LSTM cells maintain long-term memory of previous words
- **Context**: Captures context across long sequences
- **Pattern Learning**: Learns statistical patterns in text

### Preprocessing:
- Vocabulary size: 10,000 words
- Sequence length: 50 words
- Out-of-vocabulary handling with `<OOV>` token

## Limitations

- Vocabulary limited to 10,000 most frequent words
- Fixed sequence length (50 words)
- Requires substantial training data for good results
- Generation quality depends on training corpus

## Future Improvements

- Attention mechanisms
- Transformer architecture
- Larger vocabulary support
- Variable sequence length
- Fine-tuning on specific domains
- Beam search for better generation

## License

This project is provided as-is for educational and research purposes.

## Contact

For questions or contributions, please refer to the research paper explanation document for detailed technical information.
