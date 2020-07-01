from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange

def svm_loss_naive(W, X, y, reg):
    """
    Structured SVM loss function, naive implementation (with loops).

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
    #Init
    dW = np.zeros_like(W)
    loss = 0
    delta = 0
    #Basic counters
    num_classes = W.shape[1] #C
    num_train = X.shape[0] #N
    dimensions =  X.shape[1] #D

    #Main Operations: Computing the LOSS
    scores = X.dot(W) #(N,C)
    # assert scores.shape == (num_train, num_classes)
    #Get the correct values out of the scores
    correct_scores = scores[np.arange(num_train),y] #(N,1)
    #Expand these scores    
    correct_scores = np.repeat(correct_scores,num_classes) #len: N*C -> vector
    correct_scores = np.reshape(correct_scores,scores.shape) #(N,C)
    #Subtract the scores and add the margin  then max with zero
    scores_subtracted = scores - correct_scores + delta
    scores_subtracted[scores_subtracted<0] = 0 
    #Remove the correct classes from computations
    scores_subtracted[np.arange(num_train),y]=0
    #Sum the loss
    loss = np.sum(scores_subtracted)
    loss /= num_train
    #Add reg
    loss+=0.5*reg*np.sum(W*W)


    #COMPUTING THE GRAD
    #derivative of the max operation
    margins = np.zeros_like(scores_subtracted)
    margins[scores_subtracted>0]=1 

    assert margins.shape == (num_train, num_classes)

    row_sum = np.sum(margins, axis=1)   # N vector
    #margins[np.arange(num_train), y]:
        #margins of 0->499 where j==y
        #this reduces a vector of size num_train
    margins[np.arange(num_train), y] = -row_sum

    dW += X.T.dot(margins)     # (D, C)
    assert dW.shape == (dimensions, num_classes)

    #normalize and add regularizatoin
    dW/=num_train
    dW += reg * W

    #############################################################################
    # TODO:                                                                     #
    # Compute the gradient of the loss function and store it dW.                #
    # Rather that first computing the loss and then computing the derivative,   #
    # it may be simpler to compute the derivative at the same time that the     #
    # loss is being computed. As a result you may need to modify some of the    #
    # code above to compute the gradient.                                       #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****


    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    return loss, dW



def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.

    Inputs and outputs are the same as svm_loss_naive.
    """
    loss = 0.0
    dW = np.zeros(W.shape) # initialize the gradient as zero

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the structured SVM loss, storing the    #
    # result in loss.                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the gradient for the structured SVM     #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    #Init
    dW = np.zeros_like(W)
    loss = 0
    delta = 0
    #Basic counters
    num_classes = W.shape[1] #C
    num_train = X.shape[0] #N
    #Main Operations
    scores = X.dot(W) #(N,C)
    # assert scores.shape == (num_train, num_classes)

    #Get the correct values out of the scores
    correct_scores = scores[np.arange(num_train),y] #(N,1)
    #Expand these scores    
    correct_scores = np.repeat(correct_scores,num_classes) #len: N*C -> vector
    correct_scores = np.reshape(correct_scores,scores.shape) #(N,C)
    #Subtract the scores and add the margin  then max with zero
    scores_subtracted = scores - correct_scores + delta
    scores_subtracted[scores_subtracted<0] = 0 
    #Remove the correct classes from computations
    scores_subtracted[np.arange(num_train),y]=0
    #Sum the loss
    loss = np.sum(scores_subtracted)
    loss /= num_train
    #Add reg
    loss+=0.5*reg*np.sum(W*W)

    #COMPUTING THE GRAD
    #derivative of the max operation
    margins = np.zeros_like(scores_subtracted)
    margins[scores_subtracted>0]=1 

    assert margins.shape == (num_train, num_classes)

    row_sum = np.sum(margins, axis=1)   # N vector
    #margins[np.arange(num_train), y]:
        #margins of 0->499 where j==y
        #this reduces a vector of size num_train
    margins[np.arange(num_train), y] = -row_sum

    dW += X.T.dot(margins)     # (D, C)
    assert dW.shape == (dimensions, num_classes)

    #normalize and add regularizatoin
    dW/=num_train
    dW += reg * W
    
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
