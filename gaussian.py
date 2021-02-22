# Sets of Gaussians

# Import dependencies.
from sklearn import preprocessing
from sklearn.naive_bayes import GaussianNB

# Create data set.
at1 = [3.2, 5.2, 8.5, 2.3, 6.2, 1.3]
at2 = [2.1, 6.1, 1.3, 5.4, 3.1, 6.0]
at3 = [2.1, 7.5, 0.5, 2.45, 4.4, 3.35]
class_label = 'pos pos pos neg neg neg'.split()

# Encode data set into numbers.
label_encoder = preprocessing.LabelEncoder()
encoded_class_label = label_encoder.fit_transform(class_label)

# Build the classifier.
classifier = []
for i in range(len(class_label)):
	classifier.append([at1[i], at2[i], at3[i]])

# Train the model.
model = GaussianNB()
model.fit(classifier, encoded_class_label)

# Test the model. Positive = 1, Negative = 0.
user_at = []
print('Please input the attributes you want to test.')
print('Attribute 1:')
user_at.append(float(input()))
print('Attribute 2:')
user_at.append(float(input()))
print('Attribute 3:')
user_at.append(float(input()))
print('The class for', user_at, 'is:')
print(model.predict([user_at]))