# K-NN Classifier

# Import dependencies.
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier

# Create training set.
at1 = [1, 3, 3, 5]
at2 = [3, 5, 2, 2]
at3 = [1, 2, 2, 3]
labels = 'pos pos neg neg'.split()

# Encode class-labels into numbers.
label_encoder = preprocessing.LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)

# Build the classifier.
classifier = list(zip(at1, at2, at3))

# Train the model.
model = KNeighborsClassifier(n_neighbors = 3)
model.fit(classifier, encoded_labels)

# Test the model. Positive = 1, negative = 0.
test = [2, 4, 2]
if model.predict([test]) == 1:
	print('positive')
else:
	print('negative')