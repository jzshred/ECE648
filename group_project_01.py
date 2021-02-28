# K-NN classifier with TA evaluation dataset from UCI repository

# Import dependencies.
import pandas as pd
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from numpy.random import seed
from numpy.random import randint
from numpy.random import uniform

# Function to train and test the model with 1-10 nearest neighbors.
def train_and_test():
	for i in range(10):
		model = KNeighborsClassifier(n_neighbors = i + 1)
		model.fit(training_set, training_class)
		predicted_class = model.predict(testing_set)
		scores.append(metrics.accuracy_score(testing_class, predicted_class))

# Function to print test results.
def print_results():
	for i in range(10):
		print(str(i + 1) + "-NN:", scores[i])
	print()

# Import data set, name the columns, separate class labels.
dataset = pd.read_csv('tae.data', header = None)
dataset.columns = 'native instructor course summer size eval'.split()
labels = dataset.pop('eval')

# Normalize the dataset.
scaler = MinMaxScaler()
norm_dataset = pd.DataFrame(scaler.fit_transform(dataset), columns = dataset.columns)

# Split the data into training set (70%) and testing set (30%).
training_set, testing_set, training_class, testing_class = \
train_test_split(norm_dataset, labels, test_size = 0.3, random_state = 24)

# Train and test the model with 1-10 neighbors.
scores = []
train_and_test()

# Print results.
print("Accuracy rates - without added examples:")
print_results()

# -------------------------------------------------------
# Now we start adding noisy examples to the training set.
# -------------------------------------------------------

# Create a list of random Boolean values.
bool_list = randint(0, 2, len(training_set.columns))

# Multiply the Boolean list by a noise intensity level.
intensity = uniform(-0.05, 0.05)
noise = [element * intensity for element in bool_list]

# Retrieve a random example from the training set.
random_row = randint(0, len(training_set))
new_example = training_set.iloc[random_row].values.tolist()

# Add noise to the randomly chosen example.
new_example = [x + y for x, y in zip(new_example, noise)]

# Add new example to the training set.
new_series = pd.Series(new_example, index = training_set.columns)
training_set = training_set.append(new_series, ignore_index = True)
training_class[len(training_class)] = training_class[random_row]
	
# Train and test the model with 1-10 neighbors.
scores = []
train_and_test()

# Print results.
print("Accuracy rates - with 10 added noisy examples:")
print_results()