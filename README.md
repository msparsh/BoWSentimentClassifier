# Bag of Words Sentiment Classifier

This project implements a sentiment classifier using the Bag of Words approach on the IMDB movie reviews dataset. The classifier is built using TensorFlow and TensorFlow Datasets, leveraging text vectorization and a simple neural network model.

## Features
- **Text Preprocessing**:
  - Build a vocabulary of size 1000.
  - Tokenize and vectorize reviews using `TextVectorization`.

- **Model Architecture**:
  - Input layer with 1000 dimensions.
  - Dense layer with ReLU activation.
  - Dropout layer for regularization.
  - Output layer with sigmoid activation for binary classification.

- **Training**:
  - Early stopping to prevent overfitting.
  - Validation on test data.

## Steps

### Preprocessing
1. **Build Vocabulary**: Extract vocabulary from the training dataset by processing the dataset in batches of 64.
2. **Vectorize Reviews**: Convert reviews into numerical representations using `TextVectorization` to count tokens.

### Model
- A simple feedforward neural network is used for classification:
  - Dense layer with 100 units and ReLU activation.
  - Dropout layer with a rate of 0.5.
  - Output layer with 1 unit and sigmoid activation.

### Training
- The model is trained for up to 10 epochs with early stopping based on validation loss with patience of 3.

### Evaluation
- The model is evaluated on the test dataset, and loss and accuracy are reported.

## Dependencies
- TensorFlow 2.19.0
- TensorFlow Datasets

## Usage

### Install Dependencies
Run the following command to install the required libraries:
```bash
pip install tensorflow tensorflow-datasets
```

### Run the Notebook
Open the `Bag of Words.ipynb` notebook in Jupyter or VS Code and execute the cells step-by-step.

### Results
After training, the model achieves a **test accuracy** of approximately **84.8%**.

## Future Enhancements
- Remove punctuation and stopwords during preprocessing.
- Add more metrics for evaluation.

## License
This project is licensed under the Apache License Version 2.0.
