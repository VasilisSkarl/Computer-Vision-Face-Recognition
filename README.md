# BSc Thesis: Automatic Face Recognition using Computer Vision

This repository contains the code and dataset for my BSc thesis. The project focuses on building a facial recognition system using Neural Networks and Computer Vision techniques.

Repository Files:
-------

1. dataset.csv
   - Contains 300 synthetic records (100 for each identity)
   - Each record consists of 128 numerical features, structured to simulate real facial embeddings.
   - Target column: 'Identity' (Classes: 'Person_1', 'Person_2', 'Person_3')

2. recognition.py
   - Trains a Multi-Layer Perceptron (MLP) neural network to predict identities
   - The pipeline includes:
     • Data preprocessing
     • Feature normalization
     • Stratified train/test split
     • Multiclass one-hot encoding
     • Accurate evaluation and classification report
   - Tech Stack: 'pandas', 'scikit-learn', 'tensorflow'/'keras'
   - Achieves 100% accuracy on the test set

How to Run:
------------------

Prerequisites:
- Python 3.8+
- Install the required packages:
  pip install pandas scikit-learn tensorflow

Run:
---------
Open the terminal and run:
> python recognition.py

Output:
-------
- Displays the classification accuracy
- Displays the detailed classification report (precision, recall, f1-score)

Note:
---------
The dataset is synthetic, but it is structured to simulate the embeddings that would be produced by a real-world facial recognition system (e.g., FaceNet, Dlib)
