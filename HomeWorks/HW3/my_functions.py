import h5py
import numpy as np

def load_dataset(train_path, test_path):
    train_dataset = h5py.File(train_path, "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:]) # your train set labels

    test_dataset = h5py.File(test_path, "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:]) # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:]) # your test set labels

    classes = np.array(test_dataset["list_classes"][:]) # the list of classes
    
    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))
    
    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes

def sigmoid(z):
    '''
        input: z (a scalar)
        output: sigmoid(z)
    '''
    return 1/(1 + np.exp(-z))

def propagate(w, b, x, y):
    '''
        input: - w: weights, a numpy array of size (px * px * 3, 1)
               - b: bias
               - x: data of size (px * px * 3, number of examples)
               - y: label vector of size (1, number of examples)  

        output:
               - cost: cost for logistic regression
               - dw: gradient of the loss with respect to w
               - db: gradient of the loss with respect to b
    '''
    m = x.shape[1]
    
    # Forward Propagation:
    activation = sigmoid(np.dot(w.T, x) + b)
    cost = -(1/m)*(np.sum(y*np.log(activation) + (1-y)*np.log(1-activation)))

    # Backward Propagation
    gradient = {'dw': (1/m)*np.dot(x, (activation-y).T), 
                'db': (1/m)*np.sum(activation-y)}

    return gradient, cost

def optimize(w, b, x, y, num_iterations, lr):
    '''
        input: - w: weights, a numpy array of size (px * px * 3, 1)
               - b: bias
               - x: data of size (px * px * 3, number of examples)
               - y: label vector of size (1, number of examples) 
               - num_iterations: number of iterations of the optimization loop
               - lr: learning_rate, for the GD update rule
        
        output:
               - params: dictionary containing w and b
               - grads: dictionary containing the gradients of the weights and bias with respect to the cost function
               - costs: list of all the costs computed during the optimization, this will be used to plot the learning curve
    '''
    costs = []

    for i in range(num_iterations):
        gradient, cost = propagate(w, b, x, y)

        dw, db = gradient['dw'], gradient['db']
        
        w -= lr*dw
        b -= lr*db
        
        if i % 100 == 0:
            costs.append(cost)
            print('{}th iteration cost: {}'.format(i, cost))

    return {'w': w, 'b': b},\
            {'dw': dw, 'db': db},\
            costs

def predict(w, b, x):
    '''
        input: - w: weights (px * px * 3, 1)
               - b: bias
               - x: data of size (px * px * 3, number of examples)
        
        output:
               - y_pred: np array containing all predictions (0 / 1) for examples in x
    '''
    m = x.shape[1]
    y_pred = np.zeros((1, m))
    w = w.reshape(x.shape[0], 1)
    activation = sigmoid(np.dot(w.T, x) + b)

    for i in range(activation.shape[1]):
        y_pred[0,i] = 1 if activation[0, i] > 0.5 else 0

    return y_pred
