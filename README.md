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