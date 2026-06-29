# SMS-Spam-Detection-Machine-Learning
A comprehensive machine learning project, utilizing a Jupyter Notebook for the SMS Spam Collection Dataset, for classifying SMS messages as spam or ham (legitimate) using three different deep learning architectures implemented in TensorFlow and Keras.
## Overview

This project implements and evaluates three distinct neural network architectures for binary text classification (spam vs. ham):

  1. Dense Embedding Model - Simple feedforward network with embedding layer

  2. Bidirectional LSTM Model - Sequence modeling with Bi-LSTM layers

  3. Transfer Learning Model - Universal Sentence Encoder (USE) for feature extraction

## Features

  • Data Preprocessing: Cleaning, deduplication, and encoding of SMS messages
      
  • Exploratory Data Analysis: Visualizations of message characteristics and class distributions

  • Text Vectorization: Custom TextVectorization layer for tokenization

  • Three Model Architectures: Comparative analysis of different approaches

  • Comprehensive Metrics: Accuracy, Precision, Recall, and F1-Score evaluation

  • Visualization: Bar charts and line graphs for performance comparison

## Project Structure

`sms-spam-detection/
├── README.md
├── notebook.ipynb (or .py script)
├── data/
│   └── spam.csv (SMS Spam Collection Dataset)
└── requirements.txt`


## Requirements

### Python Libraries

``` python
numpy
pandas
matplotlib
seaborn
tensorflow>=2.0
tensorflow-hub
scikit-learn 
```

### Installation

#### Dependencies

``` txt
numpy>=1.21.0
pandas>=1.3.0
matplotlib>=3.4.0
seaborn>=0.11.0
tensorflow>=2.8.0
tensorflow-hub>=0.12.0
scikit-learn>=1.0.0

```
Install the Dependencies

``` bash

pip install -r requirements.txt

```

### Environment

    • Python 3.x

    • Jupyter Notebook

    • Kaggle environment (recommended) or local setup with TensorFlow 2.x

## Dataset
### Source: SMS Spam Collection Dataset

### Dataset Characteristics:

  • 5,572 SMS messages

  • Labeled as "ham" (legitimate) or "spam"

  • Encoded in Latin-1

  • Class Distribution: Approximately 86.6% ham, 13.4% spam

  • Average message length: ~16 words

  • Vocabulary size: ~3,000 unique words
  

### Format: CSV file with columns:

  • label: 'ham' (legitimate) or 'spam'

  • text: SMS message content
  

### Preprocessing Steps:

  • Removed unnecessary columns (Unnamed: 2-4)

  • Renamed columns for clarity

  • Encoded labels (ham=0, spam=1)

  • Dropped duplicate entries

  • 80/20 train-test split

## Model Architectures
1. Dense Embedding Model

```python
Input → TextVectorization → Embedding(128) → GlobalAveragePooling1D → 
Dense(32, relu) → Dense(1, sigmoid)

```

2. Bidirectional LSTM Model

```python
Input → TextVectorization → Embedding(128) 
→ Bidirectional(LSTM(64, return_sequences=True)) 
→ Bidirectional(LSTM(64)) → Flatten → Dropout(0.1) 
→ Dense(32, relu) → Dense(1, sigmoid)

```

3. Universal Sentence Encoder (Transfer Learning)

```python
Input → USE_Layer(512) → Dense(64, relu) → Dropout(0.2) → Dense(1, sigmoid)

```

## Usage
### Quick Start

  1. Clone or download the project files

  2. Install dependencies: `pip install -r requirements.txt`

  3. Prepare data: Ensure `spam.csv` is in the correct location

  4. Run the notebook/script: Execute the main Python file or Jupyter notebook


### Code Execution

```bash
# Run as Python script
python sms_spam_detection.py

# Or open in Jupyter
jupyter notebook notebook.ipynb

```


### Custom Training


```python
# Adjust epochs
history = compile_and_fit(model, epochs=10)

# Test on custom data
custom_text = np.array(["Win money now! Click here"])
prediction = model.predict(custom_text)
```

## Performance Metrics
The notebook evaluates all models on:

   • Accuracy: Overall correctness

   • Precision: Spam detection reliability

   • Recall: Spam capture rate

   • F1-Score: Balanced mean between precision and recall

## Key Insights

  • Text Vectorization: Automatically adapts to vocabulary size and sequence length
      
  • Embedding Dimension: 128 dimensions for word representations
      
  • Sequence Length: Based on average words per message (~avg
_words_len)

  • Training: Adam optimizer with binary crossentropy loss
      
  • Validation: 20% holdout set for early stopping and evaluation

## Expected Output

The script should generate:

  • Data loading statistics

  • Exploratory data analysis plots

  • Model training progress

  • Performance comparison tables

  • Visualization charts comparing all three models
  
## Results Interpretation

After training, you'll see:

  1. Performance Table: Numerical metrics for all models

  2. Bar Chart: Side-by-side comparison of metrics

  3. Line Graph: Performance trends across different metrics

### Typical Results (may vary based on random initialization):

  • Dense Embedding: ~97-98% accuracy

  • Bi-LSTM: ~97-98% accuracy

  • USE (Transfer Learning): ~98-99% accuracy

## Feature Engineering
The project includes automated feature extraction:

  • Character count: Total characters in each message

  • Word count: Number of words per message

  • Average word length: Character count divided by word count

These features are visualized to understand differences between spam and ham messages.
## Data Preprocessing Pipeline

  1. Loading: Read CSV with Latin-1 encoding

  2. Cleaning: Drop unused columns and rename for clarity

  3. Label Encoding: Convert 'ham'/'spam' to 0/1

  4. Duplicate Removal: Eliminate exact duplicate messages

  5. Text Normalization: Convert to lowercase

  6. Feature Extraction: Calculate message statistics

  7. Train-Test Split: 80/20 split with random state fixed at 42 for reproducibility

## Visualization

  1. Pie Chart: Class distribution (ham vs spam)

  2. Histograms: Feature distributions by class

  3. Heatmap: Correlation matrix of numeric features

  4. Bar Charts: Model performance comparison

  5. Line Graphs: Performance trends across metrics

## Customization

#### Modify Training Parameters

```python
# Change epochs
history = compile_and_fit(model, epochs=10)

# Adjust test size
X_train, X_test, y_train, y_test = train_test_split(
    df['text'], df['label_enc'], test_size=0.3, random_state=42
)
```
### Add New Models

You can extend the architecture comparison by adding new models to the results dictionary: 

```python
results['New Model'] = get_metrics(new_model, X_test_np, y_test_np)

```

## Notes

  • GPU Acceleration: Recommended for faster training (especially for Bi-LSTM)

  • Memory: Requires ~20GB available for large datasets

## Contributing
Feel free to:
  • Add new model architectures

  • Improve preprocessing steps

  • Add more visualization types

  • Experiment with different hyperparameters

  • Try other pre-trained embeddings (BERT, Word2Vec)

  • Add data augmentation techniques

  • Implement early stopping callbacks

## License

This notebook is provided for educational purposes. The dataset is available under CC0 license.

## Acknowledgments
  • Dataset: SMS Spam Collection Dataset (UCI Machine Learning Repository)

  • Framework: TensorFlow 2.x and Keras

  • Embedding: Google Universal Sentence Encoder (TF-Hub)

## Support

For issues or questions:

  • Check the code comments for detailed explanations

  • Review the TensorFlow and Keras documentation

  • Explore the Universal Sentence Encoder documentation on TensorFlow Hub
