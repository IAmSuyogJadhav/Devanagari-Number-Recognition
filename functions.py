import numpy as np

# ------------------ Functions
def sigmoid(z):
    """Calculates the sigmoid activation function."""

    return 1 / (1 + np.exp(-z))


def cost(x, y, w1, w2, w3, lamda=0):
    """Calculates the cost for the neural network."""
    m, n = x.shape
    a1 = np.append(np.ones(m).reshape(m, 1), x, axis=1)
    z2 = a1.dot(w1.T)
    a2 = np.append(np.ones(z2.shape[0]).reshape(z2.shape[0], 1), sigmoid(z2), axis=1)
    z3 = a2.dot(w2.T)
    a3 = np.append(np.ones(z3.shape[0]).reshape(z3.shape[0], 1), sigmoid(z3), axis=1)
    z4 = a3.dot(w3.T)
    a4 = sigmoid(z4)

    j = (-1/m)*np.sum(y.T.dot(np.log(a4)) + (1 - y).T.dot(np.log(1 - a4)))\
        + (lamda/(2*m))*(np.sum(w1[:, 1:]**2) + np.sum(w2[:, 1:]**2) + np.sum(w3[:, 1:]**2))
    return j


def train(x, y, w1, w2, w3, alpha=0.01, lamda=0):
    """Trains the neural network. Performs one pass each of forward propagation and Back propagation."""

    m, n = x.shape
    a1 = np.append(np.ones(m).reshape(m, 1), x, axis=1)
    z2 = a1.dot(w1.T)
    a2 = np.append(np.ones(z2.shape[0]).reshape(z2.shape[0], 1), sigmoid(z2), axis=1)
    z3 = a2.dot(w2.T)
    a3 = np.append(np.ones(z3.shape[0]).reshape(z3.shape[0], 1), sigmoid(z3), axis=1)
    z4 = a3.dot(w3.T)
    a4 = sigmoid(z4)
    
# ------------------ Backprop
    del4 = a4 - y
    del3 = (del4.dot(w3)) * a3 * (1 - a3)
    # THE LAST AND SECOND LAST DEL TERMS WILL ALWAYS BE LIKE AS SHOWN ABOVE
    del2 = (del3[:, 1:].dot(w2)) * a2 * (1 - a2)
    # WHEN ADDING MORE LAYERS IN FUTURE, REPEAT THE SAME THING AS ABOVE
    # FOR ALL THE DEL TERMS EXCEPT THE SECOND LAST AND LAST ONE

    w1_grad = (1/m) * (del2[:, 1:].T.dot(a1))
    w2_grad = (1/m) * (del3[:, 1:].T.dot(a2))
    # WHEN ADDING MORE LAYERS IN FUTURE, REPEAT THE SAME THING AS ABOVE
    # FOR ALL THE GRAD TERMS EXCEPT THE LAST ONE
    w3_grad = (1/m) * (del4.T.dot(a3))
    # THE LAST GRAD TERM WILL ALWAYS BE LIKE AS SHOWN ABOVE

    w1_reg = (lamda/m) * w1
    w2_reg = (lamda/m) * w2
    w3_reg = (lamda/m) * w3
    w1_reg[:, 0] = 0
    w2_reg[:, 0] = 0
    w3_reg[:, 0] = 0

# ------------------ Update Parameters
    w1 = w1 - alpha * w1_grad - w1_reg
    w2 = w2 - alpha * w2_grad - w2_reg
    w3 = w3 - alpha * w3_grad - w3_reg

    return w1, w2, w3

def predict(x, w1, w2, w3):
    """Predicts the outcome of a trained neural network for given data."""

    m, n = x.shape
    a1 = np.append(np.ones(m).reshape(m, 1), x, axis=1)
    z2 = a1.dot(w1.T)
    a2 = np.append(np.ones(z2.shape[0]).reshape(z2.shape[0], 1), sigmoid(z2), axis=1)
    z3 = a2.dot(w2.T)
    a3 = np.append(np.ones(z3.shape[0]).reshape(z3.shape[0], 1), sigmoid(z3), axis=1)
    z4 = a3.dot(w3.T)
    a4 = sigmoid(z4)
    
    return np.argmax(a4, axis=1).reshape(m, 1)
