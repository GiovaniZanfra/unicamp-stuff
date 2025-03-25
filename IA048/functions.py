import numpy as np

def gradient_descent(X, y, learning_rate=0.01, n_iterations=1000):
    """
    Performs gradient descent to minimize the cost function for a linear regression model.

    This function iteratively adjusts the coefficients of the linear model to minimize the
    difference between the predicted values and the actual values in the training set.
    
    Parameters:
    - X (numpy.ndarray): The feature matrix for the training data, where each row represents
      a sample and each column represents a feature.
    - y (numpy.ndarray): The target variable for the training data, where each element is the
      target value for one sample.
    - learning_rate (float, optional): The learning rate used to update the coefficients in each
      iteration. Defaults to 0.01.
    - n_iterations (int, optional): The number of iterations to perform the gradient descent.
      Defaults to 1000.

    Returns:
    - numpy.ndarray: The final coefficients of the linear model after performing gradient descent.
    
    The function initializes the model coefficients to zero and then iteratively updates them by
    moving in the direction of the steepest descent, as defined by the negative gradient of the
    cost function. The learning rate controls the size of the steps taken towards the minimum.
    """
    m, n = X.shape
    # Initializing the coefficients with zeros
    beta = np.zeros(n)
    
    for iteration in range(n_iterations):
        # Calculate the gradient
        gradients = -2/m * X.T.dot(y - X.dot(beta))
        # Update the coefficients
        beta = beta - learning_rate * gradients
    
    return beta

import numpy as np

def online_gradient_descent(X, y, learning_rate=0.01, n_iterations=1000):
    """
    Performs online (stochastic) gradient descent to minimize the cost function
    for a linear regression model.

    Parameters:
    - X (numpy.ndarray): The feature matrix for the training data, where each row
      represents a sample and each column represents a feature. Assumes X already
      includes an intercept term (a column of ones).
    - y (numpy.ndarray): The target variable for the training data, where each
      element is the target value for one sample.
    - learning_rate (float, optional): The learning rate used to update the
      coefficients in each iteration. Defaults to 0.01.
    - n_iterations (int, optional): The number of iterations to perform the
      gradient descent. Defaults to 1000.

    Returns:
    - numpy.ndarray: The final coefficients of the linear model after performing
      online gradient descent.
    """
    m, n = X.shape
    beta = np.zeros(n)  # Initializing the coefficients with zeros
    
    for iteration in range(n_iterations):
        for i in range(m):
            xi = X[i:i+1]
            yi = y[i:i+1]
            prediction = xi.dot(beta)
            error = yi - prediction
            gradients = -2 * xi.T.dot(error)
            beta = beta - learning_rate * gradients
    
    return beta

# Example usage:
# Assuming X and y are defined and X includes an intercept term
# Normalize or standardize X features if necessary, except the intercept column
# beta_hat_sgd = online_gradient_descent(X, y, learning_rate=0.01, n_iterations=1000)
# print(beta_hat_sgd)


