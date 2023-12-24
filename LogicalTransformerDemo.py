# Copyright (c) 2024 Ole-Christoffer Granmo

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from tmu.models.classification.vanilla_classifier import TMClassifier
import numpy as np
from time import time

number_of_features = 10
noise = 0.3

X_train = np.random.randint(0, 2, size=(5000, number_of_features), dtype=np.uint32)
Y_train = np.logical_xor(X_train[:,0], X_train[:,1]).astype(dtype=np.uint32)
Y_train = np.where(np.random.rand(5000) <= noise, 1-Y_train, Y_train) # Adds noise

X_test = np.random.randint(0, 2, size=(5000, number_of_features), dtype=np.uint32)
Y_test = np.logical_xor(X_test[:,0], X_test[:,1]).astype(dtype=np.uint32)

# Train one TM per token (feature) per class, producing a class specific token model

tms = {}
for i in range(2):
	tms[i] = []
	print("\nTraining token models for class", i, "\n")
	for j in range(number_of_features):
		# This is where you create a local perspective for each token (self-attention)

		tm = TMClassifier(10, 15, 1.1)

		# Extract prediction target from column 'j' in X_train, only looking at training examples from class 'i'
		Y_train_token = X_train[:,j][Y_train==i].reshape(-1)

		# Remove prediction target from column 'j' in X_train
		cols = np.arange(X_train.shape[1]) != j
		X_train_token = X_train[:,cols][Y_train==i]

		tm.fit(X_train_token, Y_train_token)

		# Create test data for token prediction
		Y_test_token = X_test[:,j][Y_test==i].reshape(-1)
		cols = np.arange(X_test.shape[1]) != j
		X_test_token = X_test[:,cols][Y_test==i]

		print("\tToken %d prediction accuracy: %.2f" % (j+1, 100*(tm.predict(X_test_token) == Y_test_token).mean()))

		# Store Tsetlin machine for token 'j' of class 'i'
		tms[i].append(tm)

# Perform composite classification using the individual Tsetlin machines

class_sums = np.zeros((2, Y_test.shape[0]))

# Calculate composite clause sum per class
for i in range(2):
	# Add opp clause sum scores from each token Tsetlin machine
	for j in range(number_of_features):
		# Create input to token model
		cols = np.arange(X_test.shape[1]) != j
		X_test_token = X_test[:,cols]

		Y_test_predicted_token, Y_test_scores_token = tms[i][j].predict(X_test_token, return_class_sums=True)

		# Measure ability of token model to match the example. The class with the most fitting token model gets the highest score.
		Y_test_scores_token_combined = np.where(X_test[:,j] == 1, Y_test_scores_token[:,1] - Y_test_scores_token[:,0], Y_test_scores_token[:,0] - Y_test_scores_token[:,1])

		class_sums[i] += Y_test_scores_token_combined

# The class with the largest composite class sum wins
Y_test_predicted = class_sums.argmax(axis=0)

print("\nClass prediction accuracy:", 100*(Y_test_predicted == Y_test).mean())