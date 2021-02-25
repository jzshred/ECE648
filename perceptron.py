# Perceptron Learning Algorithm

# Import dependencies.
from sklearn.linear_model import Perceptron

# Create the training set.
x1 = [1, 1, 0]
x2 = [0, 1, 0]
labels = [0, 1, 0]

# Build the classifier.
classifier = list(zip(x1, x2))

# Train the model, define maximum 40 epochs, learning rate 0.5.
model = Perceptron(max_iter = 40, eta0 = 0.5)
model.fit(classifier, labels)

# Test the model.
test1 = [1, 0]
test2 = [1, 1]
test3 = [0, 0]
test4 = [0, 1]
print(model.predict([test1]))
print(model.predict([test2]))
print(model.predict([test3]))
print(model.predict([test4]))