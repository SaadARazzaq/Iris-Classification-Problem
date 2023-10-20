# Iris-Classification-Problem

  ![image](https://github.com/SaadARazzaq/Iris-Classification-Problem/assets/123338307/29f890b4-827d-4c9a-8fa6-64162b8523a3)

  This repository contains complete report on Iris Classification along with the code for training and evaluating a Decision Tree classifier and a Random Forest classifier on the Iris dataset. The Iris dataset is a popular dataset in machine learning and consists of measurements of four features of Iris flowers: sepal length, sepal width, petal length, and petal width. The goal is to classify the Iris flowers into three different species: Iris-setosa, Iris-versicolor, and Iris-virginica.
 
## Dataset

The Iris dataset used in this project is loaded from the "Iris.csv" file.

## Data Preprocessing

The feature data in the dataset is standardized using the StandardScaler from scikit-learn. Standardization ensures that all features have zero mean and unit variance, which can improve the performance of certain machine learning algorithms.

## Decision Tree Classifier

A Decision Tree classifier is trained on the standardized training data. The trained classifier is used to make predictions on the standardized testing data, and the accuracy of the model is evaluated using the accuracy_score metric. Additionally, a visualization of the decision tree is plotted to provide insights into the decision-making process of the classifier.

## Random Forest Classifier

A Random Forest classifier is trained on the standardized training data. The Random Forest classifier consists of multiple decision trees, and the number of trees is specified by the "n_estimators" hyperparameter. The trained classifier is used to make predictions on the standardized testing data, and the accuracy of the model is evaluated using the accuracy_score metric. Furthermore, a visualization of the first six decision trees in the Random Forest is plotted to showcase the variability of the individual trees.

## Results

The accuracy of the Decision Tree classifier on the testing data is 0.90000, while the accuracy of the Random Forest classifier is 0.96667. The Random Forest classifier demonstrates higher accuracy due to its ensemble nature, which combines predictions from multiple decision trees. Results may varry on each run.
**NOTE:  MAKE SURE THAT YOU RUN THE CODE CELL WISE ON DECISION TREE FIRST AND THEN FOR RUNNING ON THE RANDOM FOREST RERUN EACH CELL AGAIN FROM START EXCEPT DECISION TREE CODE. THIS IS BECAUSE THE TRAINING AND TESTING VALUES KEPT CHANGING IF I DID NOT FOLLOW THIS METHOD AND IN THIS WAY DECISION TREE AND RANDOM FOREST GAVE ME SAME ACCURACY. THIS IS FROM MY OBSERVATIONS**

## Usage

To use the code in this repository, follow these steps:

1. Clone the repository to your local machine.
2. Open the project in Jupyter Notebook/ Google Colab.
3. Install the required dependencies.
4. Run the code cell wise.
5. Analyze the accuracy of the Decision Tree classifier and the Random Forest classifier. may differ on each run.
6. Examine the visualizations of the decision trees.

## Getting Started

To get started with this project, follow the steps below:

1. **Clone the Repository**: Start by cloning this repository to your local machine using the following command:
2. **Install Dependencies**: Make sure you have the necessary dependencies installed. The code in this project requires the following dependencies:

- NumPy
- pandas
- scikit-learn
- matplotlib

You can install these dependencies by running the following command:
```bash
!pip install numpy pandas scikit-learn matplotlib
```
3. **Load the Dataset**: The first step is to load the Iris dataset. The dataset is stored in the "Iris.csv" file, and we can load it using the pandas library. Add the following code to load the dataset:
```bash
import pandas as pd

# Load the dataset
df = pd.read_csv("Iris.csv")
```
4. **Data Preprocessing**: Once the dataset is loaded, we need to preprocess it before training our classifiers. In this step, we will encode the species names to numerical values and remove unnecessary columns. Add the following code for data preprocessing:
```bash
# Encode the species names
species_mapping = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}
df['Encoded_Species'] = df['Species'].map(species_mapping)

# Remove unnecessary columns
df.drop(columns=['Species', 'Id'], inplace=True)
```
5. **Split the Dataset**: Now, we will split the dataset into a training set and a testing set. This is done to evaluate the performance of our classifiers on unseen data. Add the following code to split the dataset:
```bash
from sklearn.model_selection import train_test_split

# Split the data into a training set and a testing set
X = df.drop(columns=['Encoded_Species']).values
y = df['Encoded_Species'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```
6. **Train a Decision Tree Classifier**: We will start by training a Decision Tree classifier on the trainingset. Add the following code to train the Decision Tree classifier:
```bash
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, plot_tree

# Create a Decision Tree classifier
dt_classifier = DecisionTreeClassifier()

# Train the classifier
dt_classifier.fit(X_train, y_train)

# Make predictions on the testing data
dt_predictions = dt_classifier.predict(X_test)

# Evaluate the accuracy of the model
dt_accuracy = accuracy_score(y_test, dt_predictions)

# Plot the decision tree
plot_tree(dt_classifier)
```
7. **Train a Random Forest Classifier**: Next, we will train a Random Forest classifier on the training set. Add the following code to train the Random Forest classifier:
```bash
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, plot_tree

# Create a Random Forest classifier
rf_classifier = RandomForestClassifier(n_estimators=100)

# Train the classifier
rf_classifier.fit(X_train, y_train)

# Make predictions on the testing data
rf_predictions = rf_classifier.predict(X_test)

# Evaluate the accuracy of the model
rf_accuracy = accuracy_score(y_test, rf_predictions)

# Plot the first six decision trees in the Random Forest
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(10, 7), dpi=300)
for i, (ax, tree) in enumerate(zip(axes.flatten(), rf_classifier.estimators_[:6])):
    ax.set_title(f"Tree {i+1}")
    plot_tree(tree, ax=ax)
plt.tight_layout()
plt.show()
```
8. **Results**: Finally, print out the accuracy of the Decision Tree classifier and the Random Forest classifier:
```bash
print(f"Decision Tree Accuracy: {dt_accuracy:.5f}")
print(f"Random Forest Accuracy: {rf_accuracy:.5f}")
```

## Contributing

Contributions to this repository are welcome. If you find any issues or have suggestions for improvements, please open an issue or submit a pull request.

## Contact

For any inquiries or questions, you can reach out to the project maintainer:

Name: [Saad Abdur Razzaq]

Email: [sabdurrazzaq124@gmail.com]

Linkedin: [Let's Connect](https://www.linkedin.com/in/saadarazzaq/)

Feel free to get in touch!

```bash
Made with 💖 by Saad
```
