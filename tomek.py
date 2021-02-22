# Import required libraries.
from sklearn import preprocessing
from imblearn.under_sampling import TomekLinks

# Create training set.
atx = [1.0, 1.0, 1.2, 4.0, 2.0, 2.5, 3.0, 3.5,
	   4.8, 4.5, 4.8, 3.8, 3.8, 4.0, 4.5, 2.5]
aty = [4.0, 7.0, 8.0, 1.0, 2.0, 6.0, 3.0, 5.5,
	   8.0, 2.0, 7.0, 10.0, 7.5, 5.5, 4.0, 9.0]
labels = (('pos ' * 9) + ('neg ' * 7)).split()

# Encode class-labels into numbers.
label_encoder = preprocessing.LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)

# Build the classifier.
classifier = list(zip(atx, aty))

# Print training set before removing Tomek Links.
print("Before removing Tomek Links:")
print(classifier)
print(len(classifier))

# Define the undersampling method and transform the dataset.
undersample = TomekLinks(sampling_strategy='all')
classifier, encoded_labels = undersample.fit_resample(classifier, encoded_labels)

# Print training set after removing Tomek Links.
print("After removing Tomek Links:")
print(classifier)
print(len(classifier))