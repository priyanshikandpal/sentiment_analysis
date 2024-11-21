# svm_demo.py

import numpy as np
from sklearn import svm

# Define the labels and samples
labels = np.array([1, 1, 2, 2])  # replace with your labels
samples = np.array([[1, 2], [1, 4], [2, 4], [2, 5]])  # replace with your samples

# Create an SVM instance
svm_instance = svm.SVC(kernel='linear', C=10)

# Train the SVM model
svm_instance.fit(samples, labels)

# Test the SVM model
test_samples = np.array([[1, 3], [2, 5]])  # replace with your test samples
predicted_labels = svm_instance.predict(test_samples)

# Print the predicted labels
print(predicted_labels)