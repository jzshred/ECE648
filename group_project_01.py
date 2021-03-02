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
def train_and_test(training_set, testing_set, training_class, testing_class):
	scores = []
	for i in range(10):
		model = KNeighborsClassifier(n_neighbors = i + 1)
		model.fit(training_set.values, training_class.values.ravel())
		predicted_class = model.predict(testing_set)
		score = metrics.accuracy_score(testing_class, predicted_class)
		scores.append(score)
	print_results(scores)

# Function to print test results.
def print_results(scores):
	for i in range(10):
		print(str(i + 1) + "-NN:", scores[i])
	print()

# Function to import data set, name columns, separate class labels.
def get_data(path,attributes):
	dataset = pd.read_csv(path, header = None)
	dataset.columns = attributes.split()
	labels = dataset.iloc[:,-1:]
	dataset = dataset.drop(columns=['eval'])
	return (dataset,labels)

# Function to normalize the dataset.
def norm_and_split(dataset,labels):
	scaler = MinMaxScaler()
	norm_dataset = pd.DataFrame(scaler.fit_transform(dataset), columns = dataset.columns)
	training_set, testing_set, training_class, testing_class = \
	train_test_split(norm_dataset, labels, test_size = 0.3, random_state = 24)
	return (training_set, testing_set, training_class, testing_class)

# Function to adjust boolean values.
def adjust_boolean(training_set, testing_set, boolean_columns):
	for column in boolean_columns:
		training_set[column] = 0.5 * training_set[column]
		testing_set[column] = 0.5 * testing_set[column]
	return (training_set, testing_set)

# Function to create a noisy training set.
def noiser(training_set, training_class, iterations, intensity):
	# Copy training set and training class.
	noisy_training_set = training_set
	noisy_training_class = training_class

	# Run the loop to add noisy examples.
	for i in range(iterations):

		# Create a list of random Boolean values.
		bool_list = randint(0, 2, len(training_set.columns))

		# Multiply the Boolean list by a noise intensity level.
		intensity = uniform(-1 * intensity, intensity)
		noise = [element * intensity for element in bool_list]

		# Retrieve a random example from the training set.
		random_row = randint(0, len(training_set))
		new_example = training_set.iloc[random_row].values.tolist()

		# Retrieve the corresponding random class from training class.
		random_data = {list(training_class.columns)[0] : training_class.iloc[random_row].values.tolist()}

		# Add noise to the randomly chosen example.
		new_example = [x + y for x, y in zip(new_example, noise)]

		# Create new examples.
		new_series = pd.Series(new_example, index = training_set.columns)
		random_class = pd.DataFrame(random_data)

		# Update noisy training set/class.
		noisy_training_set = noisy_training_set.append(new_series, ignore_index = True)
		noisy_training_class = noisy_training_class.append(random_class, ignore_index = True)

	return (noisy_training_set, noisy_training_class)


# ---------------------------------------------------------------
# Main program instructions.
# ---------------------------------------------------------------
# Import data.
# TODO: when changing the data set, values must be modified.
dataset, labels = get_data('tae.data','native instructor course summer size eval')

# Normalize and split the data.
training_set, testing_set, training_class, testing_class = norm_and_split(dataset,labels)

# Adjust boolean attributes to 0.5.
# TODO: when changing the data set, column names must be modified.
boolean_columns = 'native summer'.split()
training_set, testing_set = adjust_boolean(training_set, testing_set, boolean_columns)

# Train without noise.
print('Accuracy rates - without added examples')
train_and_test(training_set, testing_set, training_class, testing_class)

# Define noise variables.
iterations = [10, 50, 100]
intensity = [.05, .5, 1]

# Train with noise.
for i in iterations:
	for j in intensity:
		noisy_training_set, noisy_training_class = noiser(training_set, training_class, i, j)
		print('Accuracy rates - ' + str(i) + ' added examples/noise intensity ' + str(j))
		train_and_test(noisy_training_set, testing_set, noisy_training_class, testing_class)