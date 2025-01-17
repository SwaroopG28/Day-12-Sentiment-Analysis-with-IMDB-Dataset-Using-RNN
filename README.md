# Sentiment Analysis with IMDB Dataset Using RNN

## **Project Overview**
This project builds a Sentiment Analysis model using a **Recurrent Neural Network (RNN)**. The model classifies movie reviews from the IMDB dataset as either **Positive** or **Negative**. The dataset is pre-tokenized, with reviews represented as sequences of integers corresponding to words.

### **Objective**
- Develop a binary classification model to predict the sentiment of a movie review.
- Leverage **LSTM (Long Short-Term Memory)**, a type of RNN, to capture sequential dependencies in text.

---

## **Dataset**
- **Name**: IMDB Reviews Dataset
- **Source**: TensorFlow Datasets
- **Description**: 50,000 movie reviews split into:
  - 25,000 for training
  - 25,000 for testing
- **Classes**: Binary Sentiment Classification
  - Positive: 1
  - Negative: 0

---

## **Steps in the Project**

### **1. Loading the Dataset**
- **TensorFlow's IMDB dataset** is pre-tokenized with words mapped to integers.
- Restricted the vocabulary to the top 10,000 most frequent words to reduce noise.
- Example:
  - Encoded Review: `[1, 14, 22, 16, ...]`
  - Decoded Review: "the movie was absolutely wonderful"

### **2. Preprocessing the Data**
- **Padding Sequences**:
  - Since reviews vary in length, shorter sequences are padded with zeros to a fixed length (200 words).
  - Ensures uniform input shape for the RNN.

### **3. Building the RNN Model**
- **Embedding Layer**:
  - Converts word indices into dense vectors of fixed size (64 dimensions).
  - Captures semantic relationships between words.
- **LSTM Layer**:
  - Handles sequential data by remembering long-term dependencies.
  - Includes dropout to prevent overfitting.
- **Dense Layers**:
  - A hidden layer with ReLU activation for learning complex patterns.
  - Output layer with sigmoid activation for binary classification.

### **4. Training the Model**
- Optimizer: Adam (adaptive learning rate).
- Loss: Binary cross-entropy (suitable for binary classification).
- Metrics: Accuracy to evaluate performance.
- Epochs: 5 (adjustable for better results).

### **5. Evaluating the Model**
- Evaluated on the test set.
- Metrics:
  - **Accuracy**: ~85% (may vary depending on training).
  - **Loss**: Tracked to ensure proper convergence.

### **6. Visualizing Results**
- Plotted **training and validation accuracy** over epochs.
- Plotted **training and validation loss** over epochs.

### **7. Testing Custom Reviews**
- Tested the model with custom movie reviews to predict sentiment.
- Preprocessed reviews (cleaned and tokenized).
- Used the model to predict probabilities:
  - Probability > 0.5: Positive
  - Probability <= 0.5: Negative

---

## **Results**
- **Model Performance**:
  - Test Accuracy: 85.49%.
- **Predictions on Custom Reviews**:
  - "The movie was fantastic! I loved every bit of it." → **Positive**
  - "The plot was dull and boring. I wasted my time." → **Negative**


