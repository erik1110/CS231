from builtins import range
from builtins import object
import numpy as np

from cs231n.layers import *
from cs231n.layer_utils import *


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

    def __init__(self, input_dim=3*32*32, hidden_dim=100, num_classes=10,
                 weight_scale=1e-3, reg=0.0):
        """
        Initialize a new network.

        Inputs:
        - input_dim: An integer giving the size of the input
        - hidden_dim: An integer giving the size of the hidden layer
        - num_classes: An integer giving the number of classes to classify
        - dropout: Scalar between 0 and 1 giving dropout strength.
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - reg: Scalar giving L2 regularization strength.
        """
        self.params = {}
        self.reg = reg

        ############################################################################
        # TODO: Initialize the weights and biases of the two-layer net. Weights    #
        # should be initialized from a Gaussian with standard deviation equal to   #
        # weight_scale, and biases should be initialized to zero. All weights and  #
        # biases should be stored in the dictionary self.params, with first layer  #
        # weights and biases using the keys 'W1' and 'b1' and second layer weights #
        # and biases using the keys 'W2' and 'b2'.                                 #
        ############################################################################
        
        #pass
        #np.random.normal(平均數,std,輸出的shape)
        #input_dim = 5
        #hidden_dim = 50
        #num_classes = 7
        #weight_scale = 1e-3
        
        #X = (3,5)
        #y = (,7)
        self.params['W1'] = np.random.normal(0, weight_scale, (input_dim, hidden_dim))#隨機建立 5*50 的矩陣 
        self.params['b1'] = np.zeros((1, hidden_dim)) # (1,50)
        self.params['W2'] = np.random.normal(0, weight_scale, (hidden_dim, num_classes)) #隨機建立 50*7 的矩陣
        self.params['b2'] = np.zeros((1, num_classes)) #(1,7)
        
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
        
        #pass
        #用剛剛做好的RELU
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        h1, cache1 = affine_relu_forward(X, W1, b1)
        out, cache2 = affine_forward(h1, W2, b2)
        scores = out
        
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
        
        #用softmax算loss
        data_loss, dscores = softmax_loss(scores, y)
        #要L2
        reg_loss = 0.5 * self.reg * np.sum(W1*W1) + 0.5 * self.reg * np.sum(W2*W2)
        loss = data_loss + reg_loss
        
       # 用剛剛做好的RELU backward
       # Backward pass: compute gradients
        dh1, dW2, db2 = affine_backward(dscores, cache2)
        dX, dW1, db1 = affine_relu_backward(dh1, cache1)

       # W更新
        dW2 += self.reg * W2
        dW1 += self.reg * W1
        
        grads['W1'] = dW1
        grads['b1'] = db1
        grads['W2'] = dW2
        grads['b2'] = db2 
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

    def __init__(self, hidden_dims, input_dim=3*32*32, num_classes=10,
                 dropout=1, normalization=None, reg=0.0,
                 weight_scale=1e-2, dtype=np.float32, seed=None):
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
        self.normalization = normalization
        self.use_dropout = dropout != 1
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}
        self.channels = [input_dim] + hidden_dims + [num_classes]

        ############################################################################
        # TODO                                                                     #
        # Initialize the parameters of the network, storing all values in          #
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
        for idx in range(self.num_layers):
            idx_str = str(idx+1)
            in_channels = self.channels[idx]
            out_channels = self.channels[idx + 1]
            self.params['W' + idx_str] = weight_scale * np.random.randn(in_channels, out_channels)
            self.params['b' + idx_str] = np.zeros(out_channels)
            if self.normalization != None:  # batchnorm or layernorm
                if idx != self.num_layers - 1:  # exclude output layer
                    self.params['gamma' + idx_str] = np.ones(out_channels)
                    self.params['beta' + idx_str] = np.zeros(out_channels)
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # When using dropout we need to pass a dropout_param dictionary to each
        # dropout layer so that the layer knows the dropout probability and the mode
        # (train / test). You can pass the same dropout_param to each dropout layer.
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {'mode': 'train', 'p': dropout}
            if seed is not None:
                self.dropout_param['seed'] = seed

        # With batch normalization we need to keep track of running means and
        # variances, so we need to pass a special bn_param object to each batch
        # normalization layer. You should pass self.bn_params[0] to the forward pass
        # of the first batch normalization layer, self.bn_params[1] to the forward
        # pass of the second batch normalization layer, etc.
        self.bn_params = []
        if self.normalization == 'batchnorm':
            self.bn_params = [{'mode': 'train'} for i in range(self.num_layers - 1)]
        if self.normalization == 'layernorm':
            self.ln_params = [{} for i in range(self.num_layers - 1)]

        # Cast all parameters to the correct datatype
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)

    def loss(self, X, y=None):
        """
        Compute loss and gradient for the fully-connected net.
        Input / output: Same as TwoLayerNet above.
        """
        X = X.astype(self.dtype)
        mode = 'test' if y is None else 'train'

        # Set train/test mode for batchnorm params and dropout param since they
        # behave differently during training and testing.
        if self.use_dropout:
            self.dropout_param['mode'] = mode
        if self.normalization == 'batchnorm':
            for bn_param in self.bn_params:
                bn_param['mode'] = mode
        elif self.normalization == 'layernorm':
            for ln_param in self.ln_params:
                ln_param['mode'] = mode
        scores = None
        ############################################################################
        # TODO                                                                     #
        # Implement the forward pass for the fully-connected net, computing        #
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
        cache = []
        cache_dropout = []
        inputs = X
        for idx in range(self.num_layers):
            W = self.params['W' + str(idx+1)]
            b = self.params['b' + str(idx+1)]
            if idx == self.num_layers - 1:  # output layer need no activation
                scores, cache_layer = affine_forward(inputs, W, b)
            else:
                if self.normalization == 'batchnorm':
                    gamma = self.params['gamma' + str(idx+1)]
                    beta = self.params['beta' + str(idx+1)]
                    outputs, cache_layer = affine_bn_relu_forward(inputs, W, b, gamma, beta, self.bn_params[idx])
                elif self.normalization == 'layernorm':
                    gamma = self.params['gamma' + str(idx+1)]
                    beta = self.params['beta' + str(idx+1)]
                    outputs, cache_layer = affine_ln_relu_forward(inputs, W, b, gamma, beta, self.ln_params[idx])
                else:
                    outputs, cache_layer = affine_relu_forward(inputs, W, b)
                if self.use_dropout:
                    outputs, cache_dropout_layer = dropout_forward(outputs, self.dropout_param)
                    cache_dropout.append(cache_dropout_layer)
                inputs = outputs
            cache.append(cache_layer)
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If test mode return early
        if mode == 'test':
            return scores

        loss, grads = 0.0, {}
        ############################################################################
        # TODO                                                                     #
        # Implement the backward pass for the fully-connected net. Store the loss  #
        # in the loss variable and gradients in the grads dictionary. Compute data #
        # loss using softmax, and make sure that grads[k] holds the gradients for  #
        # self.params[k]. Don't forget to add L2 regularization!                   #
        #                                                                          #
        # When using batch/layer normalization, you don't need to regularize the   #
        # scale and shift parameters.                                              #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        loss, dout = softmax_loss(scores, y)
        for idx in range(self.num_layers - 1, -1, -1):
            idx_str = str(idx+1)
            loss += 0.5 * self.reg * np.sum(self.params['W' + idx_str] ** 2)
            if idx == self.num_layers - 1:
                dout, dW, db = affine_backward(dout, cache[idx])
            else:
                if self.use_dropout:
                    dout = dropout_backward(dout, cache_dropout[idx])
                if self.normalization == 'batchnorm':
                    dout, dW, db, dgamma, dbeta = affine_bn_relu_backward(dout, cache[idx])
                    grads['gamma' + idx_str] = dgamma
                    grads['beta' + idx_str] = dbeta
                elif self.normalization == 'layernorm':
                    dout, dW, db, dgamma, dbeta = affine_ln_relu_backward(dout, cache[idx])
                    grads['gamma' + idx_str] = dgamma
                    grads['beta' + idx_str] = dbeta
                else:
                    dout, dW, db = affine_relu_backward(dout, cache[idx])
            grads['W' + idx_str] = dW + self.reg * self.params['W' + idx_str]
            grads['b' + idx_str] = db
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads