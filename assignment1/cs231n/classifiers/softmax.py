from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    #Loop on each example
    num_train = X.shape[0]
    num_classes = W.shape[1]
    #FOR MORE STABILITY
    shiftX = X - np.max(X)
    for i in range (num_train):
        scores = shiftX[i].dot(W) #(1,C)
        exp_scores = np.exp(scores) #(1,C)
        exp_sum_scores =  np.sum(exp_scores)
        loss_i = exp_scores[y[i]] /exp_sum_scores
        loss -= np.log(loss_i)
        for j in range(num_classes):
            #shiftX[i] -> (1,D)
            #dw[:,J] -> (D,1)
            #but both are vectors, so no worries
            dW[:,j] += (exp_scores[j]/exp_sum_scores) * shiftX[i]

        dW[:,y[i]] -= shiftX[i]

    loss /= num_train
    loss += 0.5*reg * np.sum(W*W)

    dW /= num_train
    dW += reg * W
    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    #scores = X.dot(W) #(N,C)
    
    #Loop on each example
    num_train = X.shape[0]
    num_classes = W.shape[1]
    #FOR MORE STABILITY
    shiftX = X - np.max(X)

    #Compute Loss
    scores = shiftX.dot(W) #(N,C)
    exp_scores = np.exp(scores) #(N,C)

    exp_sum_scores =  np.sum(exp_scores, axis = 1) #(N,1)
    exp_scoes_at_y = exp_scores[np.arange(num_train),y] #(N,1)
    loss_i = exp_scoes_at_y / exp_sum_scores
    loss_i_log = -np.log(loss_i)
    loss = np.sum(loss_i_log) / num_train
    
    #Compute gradient
    #first for each row at dw
        #divide the exp_score for each class by the sum of that row
    exp_sum_repeated = np.reshape(np.repeat(exp_sum_scores,num_classes),exp_scores.shape) #(N,C)

    #second, divide each score by its sum
    exp_score_divided = exp_scores / exp_sum_repeated

    #then, subtract one at the values of y
    exp_score_divided[np.arange(num_train),y] -= 1

    #then multiply by X ->(N,D) and dw (D,C) and exp_score_subtracted (N,C)
    dW = shiftX.T.dot(exp_score_divided)

    #lastly, div by num of training and add regularization
    dW/=num_train
    dW+=reg*W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
