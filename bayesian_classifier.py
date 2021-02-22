# Bayesian Classifier with discrete attributes

# Import dependencies.
from sklearn import preprocessing
from sklearn.naive_bayes import CategoricalNB

# Create data set.
shape = 'circle circle triangle circle square circle circle square \
		 triangle circle square triangle'.split()

crust_size = 'thick thick thick thin thick thick thick thick thin \
			  thick thick thick'.split()

crust_shade = 'gray white dark white dark white gray white gray dark \
			   white white'.split()

filling_size = 'thick thick thick thin thin thin thick thick thin \
				thick thick thick'.split()

filling_shade = 'dark dark gray dark white dark white gray dark \
				 white dark gray'.split()

example_class = 'pos pos pos pos pos pos neg neg neg neg neg \
				 neg'.split()

# Debug: check size of attributes and class.
# print('Attribute 1:', len(shape))
# print('Attribute 2:', len(crust_size))
# print('Attribute 3:', len(crust_shade))
# print('Attribute 4:', len(filling_size))
# print('Attribute 5:', len(filling_shade))
# print('Class:', len(example_class))

# Encode data set into numbers.
label_encoder = preprocessing.LabelEncoder()
encoded_shape = label_encoder.fit_transform(shape)
encoded_crust_size = label_encoder.fit_transform(crust_size)
encoded_crust_shade = label_encoder.fit_transform(crust_shade)
encoded_filling_size = label_encoder.fit_transform(filling_size)
encoded_filling_shade = label_encoder.fit_transform(filling_shade)
encoded_example_class = label_encoder.fit_transform(example_class)

# Build the classifier.
classifier = []
for i in range(len(encoded_shape)):
	classifier.append([encoded_shape[i], encoded_crust_size[i], \
		encoded_crust_shade[i], encoded_filling_size[i], \
		encoded_filling_shade[i]])

# Train the model.
model = CategoricalNB()
model.fit(classifier, encoded_example_class)

# Test the model. Positive = 1, Negative = 0.

# square, thick, gray, thin, white
print(model.predict([[1, 0, 1, 1, 2]]))

# circle, thick, gray, thin, white
print(model.predict([[0, 0, 1, 1, 2]]))