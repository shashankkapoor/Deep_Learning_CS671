README :
-------------
Perceptron model for Multi-class Classification Problem
This code is an implementation of the Perceptron algorithm for a multi-class classification problem. In this case, the dataset contains three classes, and each class has two features. The data for each class is stored in a text file with two columns separated by a space.

--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
DEFAULT VALUE CONSIDERED
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
No of epochs considered : 10000
Value of  weights taken  :[0.4,0.4,0.4] for Classification task
                          [0.4,0.4] for univariate regression task
                          [0.4,0.4,0.4] for univariate regression task

--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Dataset
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
The paths of the text files containing the data for each class are stored in the following variables:

class_1
class_2
class_3
The data from these files is loaded using np.loadtxt() function. The data is then split into 70% training and 30% testing data for each class using np.split(). The training data is used to train the Perceptron model, while the testing data is used to evaluate the performance of the model.

--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Training the model-- 
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
The function training_perceptron()  trains the model on the training data. It takes the following arguments:

n_epoch :  The number of epochs to run the training for.
training_data:The training data for a particular class.
weights:  The initial weights for the model.
target_training_label: The target labels for the training data.
learning_factor: The learning rate for the Perceptron algorithm.

The function runs the training for the specified number of epochs, and updates the weights after processing each training example. It returns two values:
Avg_error_list: A list of the average errors for each epoch.
weights:The final weights of the model.

--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Plotting the decision boundary
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
The function plot_decision_regions() is used to plot the decision boundary for the Perceptron model. It takes the following arguments:

X:     The input features.
y:       The target labels.
weights:    The weights of the trained model.
markers:    The markers to use for each class.
colors:      The colors to use for each class.
test_idx:  The indices of the testing data.
resolution: The resolution of the meshgrid.

The function first creates a meshgrid with the specified resolution, and then predicts the class of each point in the grid using the trained model. It then uses plt.contourf() to plot the decision boundary.

--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Visualizing the dataset
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
The code includes a function that visualizes the dataset. It plots the training and testing data for each class on two subplots of a single figure.


--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Example usage
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
An example usage of the code is given in the main program. It trains the Perceptron model on the training data for each class, and then evaluates the model on the testing data. It then plots the decision boundary and the dataset.

--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Decision Region Plot for Class1-Class2-Class3
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
This code provides a decision region plot for three classes using a linear classifier. The code requires the following inputs:

train_c1_c2:  an array of the training samples for classes 1 and 2
train_c2_c3:    an array of the training samples for classes 2 and 3
weights_c1_c2:the weights for the linear classifier between classes 1 and 2
weights_c1_c3:   the weights for the linear classifier between classes 1 and 3

The code then generates a decision region plot for the three classes. The plot shows the decision boundaries between the classes and the regions where each class dominates. The markers and colors for each class are defined as follows:

To use this code, follow these steps:

Define the inputs: train_c1_c2, train_c2_c3, weights_c1_c2, and weights_c1_c3.
Copy and paste the code into a Python script or Jupyter notebook.
The decision region plot will be displayed. The plot shows the training samples for each class as well as the decision boundaries between the classes.
Create a new feature matrix X by vertically stacking the training data from three different classes and deleting the 3rd column from it using np.delete().
Create a target vector y by concatenating 0s for the first class, 1s for the second class, and 2s for the third class.
Assign weights to two different pairs of classes.
Create a color map using ListedColormap.
Define the minimum and maximum values for the two features and create a meshgrid using np.meshgrid.
Create an array Z using np.array() and insert 1s as a new column to it.
Calculate the predictions for the test data using the dot product of test data and weights for both pairs of classes and reshape the predictions into the shape of xx1 using np.reshape().
Plot the decision boundary using ax.contourf() and plot the training data points using ax.scatter().
Display the plot using plt.show().

--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Confusion Matrix
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Confusion matrix using the test data and true labels.
Note : Only Confusion Matrix is  imported from sklearn.metrics. 

Test_Data: It is a numpy array created by concatenating the test data from three different classes (test_c1, test_c2, and test_c3) along the first axis. The first axis represents the samples, while the other axes represent the features of the samples.
True_labels:It is a numpy array created by concatenating the true labels of the test data from the three different classes (test_c1, test_c2, and test_c3) along the first axis. The shape of this array is the same as the number of samples in the Test_Data array.
np.full(shape, fill_value): This function creates a numpy array of the specified shape and fills it with the specified value.
axis=0:  This parameter is used with the np.concatenate() function to specify the axis along which the arrays will be concatenated.
Test_Data.shape: This attribute of the Test_Data array returns a tuple of the dimensions of the array.
True_labels.shape: This attribute of the True_labels array returns a tuple of the dimensions of the array.












