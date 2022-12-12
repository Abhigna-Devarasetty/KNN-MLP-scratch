# vdevaras-a4

## Part 1: K-Nearest Neighbors Classification

### Steps Followed to complete the KNN Classification algorithm

1. At First, we take the train and test values (X,y) and assign them as self variables(so that it can be used anywhere in the class)
2. Then once the fit is done, we move to predict the classification type for the test data.
3. While predicting the correct classification for the data, we calculate the distance the new point and all the other points. From that points we consider top k points for classification
4. k can be passed as a parameter or default value of k here is 5
5. For calculating the distance, we have taken 2 methods Manhattan and Euclidean distance.
6. Once we find out the distance from all the points, we sort the data points based on the distance
7. Top k values are taken from sorted list and the most occuring classification is being considered and the test point is moved to that classification

## Part 2: Multilayer Perceptron Classification

### Steps Followed to complete the Multilayer Perceptron Classification

1. At First, after dividing the train and test data set. we perform the one hot encoding for the output values or to which cluster it belongs
2. In one hot encoding, we calculate the the new classification for the given y values.
3. Once the one hot encoding is done. We assign weights for each perceptron randomly based on the features given
4. Once the weights matrix is created, them we randomly create the bias values for both hidden and output layer
5. After creating the bias and weights, we pass the train data, weights to activation function and these activation function.
6. The activation function by passed in paraments, by default sigmoid is considered when no activation function name is being passed.
7. Once we have the values for the output layer, we pass it to the cross entropy function which is used to back track and find the values for y_pred
8. In predict function, We repeat the same steps that we have considered while train the data and return the predicted values for given test dataset