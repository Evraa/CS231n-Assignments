from builtins import range
from builtins import object
import numpy as np

from ..layers import *
from ..layer_utils import *


class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network with ReLU nonlinearity and
    softmax loss that uses a modular layer design. We assume an input dimension
    of D, a hidden dimension of H, and perform classification over C classes.

    The architecure should be affine - relu - affine - softmax.

    Note that this class does not implement gradient descent; instead, it
    will interact with a separate Solver object that is responsible for running
    optimization.

    The learnable parameters of the model are stored in the dictionary
    self.params that maps parameter names to numpy arrays.
    """

    def __init__(
        self,
        input_dim=3 * 32 * 32,
        hidden_dim=100,
        num_classes=10,
        weight_scale=1e-3,
        reg=0.0,
    ):
        """
        Initialize a new network.

        Inputs:
        - input_dim: An integer giving the size of the input
        - hidden_dim: An integer giving the size of the hidden layer
        - num_classes: An integer giving the number of classes to classify
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - reg: Scalar giving L2 regularization strength.
        """
        self.params = {}
        self.reg = reg

        ############################################################################
        # TODO: Initialize the weights and biases of the two-layer net. Weights    #
        # should be initialized from a Gaussian centered at 0.0 with               #
        # standard deviation equal to weight_scale, and biases should be           #
        # initialized to zero. All weights and biases should be stored in the      #
        # dictionary self.params, with first layer weights                         #
        # and biases using the keys 'W1' and 'b1' and second layer                 #
        # weights and biases using the keys 'W2' and 'b2'.                         #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        b1 = np.zeros([hidden_dim])
        b2 = np.zeros([num_classes])
        W1 = np.random.normal(0, weight_scale,[input_dim, hidden_dim])
        W2 = np.random.normal(0, weight_scale,[hidden_dim, num_classes])

        assert W1.shape == (input_dim, hidden_dim)
        assert W2.shape == (hidden_dim, num_classes)

        self.params['W1'] = W1
        self.params['W2'] = W2
        self.params['b1'] = b1
        self.params['b2'] = b2
        
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

    def loss(self, X, y=None):
        """
        Compute loss and gradient for a minibatch of data.

        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
          scores[i, c] is the classification score for X[i] and class c.

        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
          names to gradients of the loss with respect to those parameters.
        """
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the two-layer net, computing the    #
        # class scores for X and storing them in the scores variable.              #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        L1, L1_cache = affine_forward(X, self.params['W1'], self.params['b1'])
        L1_ReLu, ReLU_cahce = relu_forward(L1)
        L2, L2_cache = affine_forward(L1_ReLu, self.params['W2'], self.params['b2'])
        #No Non-linear activation for the last layer
        scores = copy.deepcopy(L2)
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If y is None then we are in test mode so just return scores
        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the two-layer net. Store the loss  #
        # in the loss variable and gradients in the grads dictionary. Compute data #
        # loss using softmax, and make sure that grads[k] holds the gradients for  #
        # self.params[k]. Don't forget to add L2 regularization!                   #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        loss, derv_soft = softmax_loss(scores, y)
        #add regularization
        W1 = self.params['W1']
        W2 = self.params['W2']
        
        loss += 0.5*self.reg*(np.sum(W1*W1) + np.sum(W2*W2))

        #GRAD

        dx2, dw2, db2 = affine_backward(derv_soft, L2_cache)

        dReLU1 = relu_backward(dx2, ReLU_cahce)

        dx1, dw1, db1 = affine_backward(dReLU1, L1_cache)

        #Add reg and store
        grads['W2'] = dw2 + self.reg*self.params['W2']
        grads['W1'] = dw1 + self.reg*self.params['W1']
        
        grads['b1'] = db1
        grads['b2'] = db2
        
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads


class FullyConnectedNet(object):
    """
    A fully-connected neural network with an arbitrary number of hidden layers,
    ReLU nonlinearities, and a softmax loss function. This will also implement
    dropout and batch/layer normalization as options. For a network with L layers,
    the architecture will be

    {affine - [batch/layer norm] - relu - [dropout]} x (L - 1) - affine - softmax

    where batch/layer normalization and dropout are optional, and the {...} block is
    repeated L - 1 times.

    Similar to the TwoLayerNet above, learnable parameters are stored in the
    self.params dictionary and will be learned using the Solver class.
    """

    def __init__(
        self,
        hidden_dims,
        input_dim=3 * 32 * 32,
        num_classes=10,
        dropout=1,
        normalization=None,
        reg=0.0,
        weight_scale=1e-2,
        dtype=np.float32,
        seed=None,
    ):
        """
        Initialize a new FullyConnectedNet.

        Inputs:
        - hidden_dims: A list of integers giving the size of each hidden layer.
        - input_dim: An integer giving the size of the input.
        - num_classes: An integer giving the number of classes to classify.
        - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=1 then
          the network should not use dropout at all.
        - normalization: What type of normalization the network should use. Valid values
          are "batchnorm", "layernorm", or None for no normalization (the default).
        - reg: Scalar giving L2 regularization strength.
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - dtype: A numpy datatype object; all computations will be performed using
          this datatype. float32 is faster but less accurate, so you should use
          float64 for numeric gradient checking.
        - seed: If not None, then pass this random seed to the dropout layers. This
          will make the dropout layers deteriminstic so we can gradient check the
          model.
        """
        # print ("Initializing your model....")
        self.normalization = normalization
        self.use_dropout = dropout != 1
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}
        ############################################################################
        # TODO: Initialize the parameters of the network, storing all values in    #
        # the self.params dictionary. Store weights and biases for the first layer #
        # in W1 and b1; for the second layer use W2 and b2, etc. Weights should be #
        # initialized from a normal distribution centered at 0 with standard       #
        # deviation equal to weight_scale. Biases should be initialized to zero.   #
        #                                                                          #
        # When using batch normalization, store scale and shift parameters for the #
        # first layer in gamma1 and beta1; for the second layer use gamma2 and     #
        # beta2, etc. Scale parameters should be initialized to ones and shift     #
        # parameters should be initialized to zeros.                               #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        first_dim = input_dim
        for i in range (self.num_layers-1):
            b_i = "b" + str(i+1)
            self.params[b_i] = np.zeros([hidden_dims[i]])
            w_i = "W" + str(i+1)
            self.params[w_i] = np.random.normal(0, weight_scale,[first_dim, hidden_dims[i]])
            first_dim = hidden_dims[i]

            if self.normalization == "batchnorm":
                self.params['beta' + str(i+1)] = np.zeros([hidden_dims[i]])
                self.params['gamma' + str(i+1)] = np.ones([hidden_dims[i]])

        #Last layer
        b_last = "b" + str(self.num_layers)
        w_last = "W" + str(self.num_layers)

        self.params[b_last] = np.zeros([num_classes])
        self.params[w_last] = np.random.normal(0, weight_scale,[first_dim, num_classes])
        
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # When using dropout we need to pass a dropout_param dictionary to each
        # dropout layer so that the layer knows the dropout probability and the mode
        # (train / test). You can pass the same dropout_param to each dropout layer.
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {"mode": "train", "p": dropout}
            if seed is not None:
                self.dropout_param["seed"] = seed

        # With batch normalization we need to keep track of running means and
        # variances, so we need to pass a special bn_param object to each batch
        # normalization layer. You should pass self.bn_params[0] to the forward pass
        # of the first batch normalization layer, self.bn_params[1] to the forward
        # pass of the second batch normalization layer, etc.
        self.bn_params = []
        if self.normalization == "batchnorm":
            self.bn_params = [{"mode": "train"} for i in range(self.num_layers - 1)]
        if self.normalization == "layernorm":
            self.bn_params = [{} for i in range(self.num_layers - 1)]

        # Cast all parameters to the correct datatype
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)

        # print ("Init your model...successfully")


    def loss(self, X, y=None):
        """
        Compute loss and gradient for the fully-connected net.

        Input / output: Same as TwoLayerNet above.
        """
        # print ("Computing the loss")
        X = X.astype(self.dtype)
        mode = "test" if y is None else "train"
        # Set train/test mode for batchnorm params and dropout param since they
        # behave differently during training and testing.
        if self.use_dropout:
            self.dropout_param["mode"] = mode
        if self.normalization == "batchnorm":
            for bn_param in self.bn_params:
                bn_param["mode"] = mode
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the fully-connected net, computing  #
        # the class scores for X and storing them in the scores variable.          #
        #                                                                          #
        # When using dropout, you'll need to pass self.dropout_param to each       #
        # dropout forward pass.                                                    #
        #                                                                          #
        # When using batch normalization, you'll need to pass self.bn_params[0] to #
        # the forward pass for the first batch normalization layer, pass           #
        # self.bn_params[1] to the forward pass for the second batch normalization #
        # layer, etc.                                                              #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        scores_cache = {}
        bn_cache = {}
        do_cache = {}
        #Caclculate regularization : np.sum(W1*W1) + np.sum(W2*W2)
        reg_sum = 0
        #A: activation
        actv = copy.deepcopy(X)
        for i in range (self.num_layers - 1):
            L, L_cache = affine_forward(actv, self.params['W'+str(i+1)], self.params['b'+str(i+1)])
            if self.normalization == "batchnorm":
                L_bn, bn_cache['bn_cache'+str(i+1)]= batchnorm_forward(L, \
                    self.params['gamma'+str(i+1)], self.params['beta'+str(i+1)], self.bn_params[i])
                L_ReLu, ReLU_cahce = relu_forward(L_bn)
            else:
                L_ReLu, ReLU_cahce = relu_forward(L)

            if self.use_dropout:
                L_do, do_cache['do_cache'+str(i+1)] = dropout_forward(L_ReLu, self.dropout_param)
                L_ReLu = L_do
            scores_cache["L_cache_"+str(i+1)] = L_cache
            scores_cache["ReLU_cahce_"+str(i+1)] = ReLU_cahce
            this_W = self.params['W'+str(i+1)]
            reg_sum += np.sum(this_W*this_W)
            actv = copy.deepcopy(L_ReLu)
            
        L_final, L_cache_final = affine_forward(L_ReLu, \
            self.params['W'+str(self.num_layers)], self.params['b'+str(self.num_layers)])
        
        scores_cache["L_cache_"+str(self.num_layers)] = L_cache_final
        this_W = self.params['W'+str(self.num_layers)]
        reg_sum += np.sum(this_W*this_W)
        scores = np.copy(L_final)
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################
        # If test mode return early
        if mode == "test":
            # print ("test: your score is: ",scores)
            return scores

        loss, grads = 0.0, {}
        ############################################################################
        # TODO: Implement the backward pass for the fully-connected net. Store the #
        # loss in the loss variable and gradients in the grads dictionary. Compute #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        #                                                                          #
        # When using batch/layer normalization, you don't need to regularize the scale   #
        # and shift parameters.                                                    #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ###########################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        loss, derv_soft = softmax_loss(scores, y)
        loss += 0.5*self.reg*(reg_sum)
        # print ("Done computing the loss: ",loss)

        #GRAD
        L_cache_final = scores_cache["L_cache_"+str(self.num_layers)]

        dx_final, dw_final, db_final = affine_backward(derv_soft, L_cache_final)
    
        grads['W'+str(self.num_layers)] = dw_final + self.reg*self.params['W'+str(self.num_layers)]
        grads['b'+str(self.num_layers)] = db_final
        for i in reversed(range(1,self.num_layers)):
            #loops L-1,L-2,...1
            #L = 3 -> 2,1
            #Relu
            #affine

            if self.use_dropout:
                do_dx_final = dropout_backward(dx_final, do_cache["do_cache"+str(i)])
                dx_final = do_dx_final
            ReLU_cahce = scores_cache["ReLU_cahce_"+str(i)]
            dReLU = relu_backward(dx_final, ReLU_cahce)
            
            if self.normalization == "batchnorm":
                dReLU_bn, dgamma, dbeta = batchnorm_backward_alt(dReLU, bn_cache['bn_cache'+str(i)])
                grads['beta' + str(i)] = dbeta
                grads['gamma' + str(i)] = dgamma
                L_cache = scores_cache["L_cache_"+str(i)]
                dx, dw, db = affine_backward(dReLU_bn, L_cache)

            else:
                L_cache = scores_cache["L_cache_"+str(i)]
                dx, dw, db = affine_backward(dReLU, L_cache)

            grads['W'+str(i)] = dw + self.reg*self.params['W'+str(i)]
            grads['b'+str(i)] = db

            dx_final = np.copy (dx)

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads


