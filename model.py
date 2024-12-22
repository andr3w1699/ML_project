# import of libraries 
import numpy as np 
import time 
import pickle
import copy

"""
Class NeuralNetwork provides the implementation of a simple neural network 

"""
class NeuralNetwork():
    
    """
    List of supported activation functions:
        - sigmoid, tanh, relu & its variants (leaky-relu, ELU), linear (identity)
    
    To extend this list --> add the definitions here below and remember to modify 
                        the function "initializeActivationFunctions" to incorporate them
    """

    def sigmoid(self, x, a=1, derivative =False):
        if(derivative):
            exp_term = np.exp(-a * x)
            return a * exp_term / (1 + exp_term)**2
        return 1/(1+np.exp(-a*x))
    
    def relu(self, x, derivative =False):
        if(derivative):
            return np.where(x > 0, 1, 0)
        return np.maximum(0,x)

    def linear(self, x, derivative =False):
        if derivative:
            return np.ones_like(x)
        return x
    
    def tanh(self, x, a=1, derivative=False):
        if derivative:
            tanh_x = np.tanh(a * x)
            return a * (1 - tanh_x**2)
        
        return np.tanh(a * x)
    
    def leaky_relu(self, x, alpha=0.01, derivative=False):
        if derivative:
            return np.where(x > 0, 1, alpha)
        return np.where(x > 0, x, alpha * x)

    def elu(self, x, alpha=1.0, derivative=False):
        if derivative:
            return np.where(x > 0, 1, alpha * np.exp(x))
        return np.where(x > 0, x, alpha * (np.exp(x) - 1))

    
    #w_{i,j}: arrow from unit i to unit j 
    @staticmethod
    def random_matrix(rows, columns, min_val, max_val):
        return np.random.uniform(low=min_val, high=max_val, size=(rows, columns))
         
    def getRandomWeights(self) :

        weightMatricesList = []
        for i in range(self.numberOfLevels): 
            weightMatricesList.append(self.random_matrix(self.units_for_levels[i]+1, self.units_for_levels[i+1], -0.05, 0.05))
        return weightMatricesList

    def getXavierWeights(self):

        weightMatricesList = []
        for i in range(self.numberOfLevels): 
            weightMatricesList.append(self.random_matrix(self.units_for_levels[i]+1, self.units_for_levels[i+1], -1/np.sqrt(self.units_for_levels[i]), 1/np.sqrt(self.units_for_levels[i])))
        return weightMatricesList


    """
    Initialization of the weight matrix based on Mode parameter
    
    ****
    initialization of the weight matrix to random small values 
    we need a list of matrices, one for each layer, 
    each matrix column represents the weights in input to a single unit of that level 
    matrix for level l in position i,j  has the weight from unit i to unit j
    the matix is m x n where m is the number of input unit and n is the number of the unit of that level
    ****
    Args:
        mode: Random for random small values, Xavier to use FanIn

    Returns:    
        A list of matrices, one for each layer, each matrix column represents the weight in input for a single unit of that level  
        matrix for level l in position i,j  has the weight from unit i to unit j
        the matix is m x n where m is the number of input unit and n is the number of the unit of that level
        
    """
    def initalizeWeightMatrix(self, mode):

        match mode:
            case "random" :
                return self.getRandomWeights()
            case "xavier" :
                return self.getXavierWeights()


    @staticmethod
    def print_matrices_fancy(list_to_print):
        for idx, matrix in enumerate(list_to_print):
            print(f"Matrix {idx + 1}:")
            for row in matrix:
                formatted_row = " | ".join(f"{elem:6.2f}" for elem in row)
                print(f"| {formatted_row} |")
            print("-" * (len(matrix[0]) * 10))  # Separator between matrices


    """
        initializeActivationFunction is the function that initialize the list of 
        activation  functions for each level, by translating the list of string 
        in a list of function 

        Args:
            activation: list of strings with the names of activation functions for each level
            
        Modify:
            It inserts in self.activation the right activation function for each level
    """
    def initializeActivationFunctions(self, activation):

        # declare an empty list of activation functions
        self.activation = []
        for levels in range(len(activation)):
            if activation[levels] == 'sigmoid':
                self.activation.append(self.sigmoid)
            elif  activation[levels] == 'relu':
                self.activation.append(self.relu)
            elif  activation[levels] == 'linear':
                self.activation.append(self.linear)
            elif  activation[levels] == 'tanh':
                self.activation.append(self.tanh)
            elif  activation[levels] == 'leaky-relu':
                self.activation.append(self.leaky_relu)
            elif  activation[levels] == 'elu':
                self.activation.append(self.elu)

    """
    Constructor: assign to self the values passed in the fields 
                 of the object "mdlParams"
    """    
    def __init__(self, mdlParams):

        # true = enable variable learning rate, false otherwise
        self.VariableLROption = mdlParams.VariableLROption
        # list of strings with the activation function for each levels
        self.activationListName = mdlParams.activation
        # parameters for variable learning rate
        # inital learning rate 
        self.eta0 = mdlParams.eta0 
        # final learning rate 
        self.eta_tau = mdlParams.eta_tau 
        # after tau epochs the learning rate is fixed to eta_tau
        self.tau = mdlParams.tau 

        # Thikonov Regularization coefficient 
        self.lambda_reg = mdlParams.lambda_reg

        # momentum coefficient 
        self.alpha = mdlParams.alpha

        # list of integer each integer position i represents the number of unit at level i 
        self.units_for_levels = mdlParams.units_for_levels
        # number of levels of the neural network 
        # e.g. [#input_unit, #hidden_unit, #output_unit] --> levels = 2
        self.numberOfLevels = len(mdlParams.units_for_levels)-1

        # type of task (classification or regression)
        self.task = mdlParams.task

        # initialize the list of activation functions 
        # transleting the list of string in a list of functions
        self.initializeActivationFunctions(mdlParams.activation)

        # true --> monitor the validation errror over training 
        #      --> we will use a given list of Weight matrices to do the training 
        self.validationErrorCheck = mdlParams.validationErrorCheck

        # if validationErrorCheck :
        #    --> initialize self.listOfWeightMatrices with a list of weight matrices passed in input
        if self.validationErrorCheck == True : 
            self.listOfWeightMatrices = copy.deepcopy(mdlParams.weights)


    """
    feedForeward: function that computes the feedForward phase, starting from an input matrix
                and using a list of weight Matrices

    Args:
        inputX: inputX is an input Matrix 
        listOfWeights: list of matrices with weights, one for each level
    
    Modify:
        At the end of feedForeward I should have as many elements as number of levels in both listOfHiddenRepr and listOfNet
        0,.., level-1 

    Returns:
        Output of the NeuralNetwork
    """
    def feedForeward (self, inputX, listOfWeights):

        localInputX = inputX
        # Aggiungere una colonna di 1 all'inizio
        localInputX = np.hstack((np.ones((inputX.shape[0], 1)), inputX))
        self.listOfHiddenRepr = []
        self.listOfNet = []

        # for all the levels of the neural network 
        for level in range(self.numberOfLevels):
            # compute the net 
            localInputX = np.matmul(localInputX, listOfWeights[level])
            # save the net 
            self.listOfNet.append(localInputX)
            # compute the hidden units' outputs 
            localInputX = self.activation[level](localInputX)
            if(level < self.numberOfLevels-1):
                localInputX = np.hstack((np.ones((localInputX.shape[0], 1)), localInputX))
            self.listOfHiddenRepr.append(localInputX)
 
         
        return localInputX; 

    """
    function backPropagate

    Args:
        x: x is the input matrix
        y: y is target matrix
        o: o is the output matrix

    Returns:
        Returns 2 elements:
            - matrix with grad output
            - list of matrices with grad hidden
    """
    def backPropagate(self, x, y, o):
        
        error = y - o

        net_k = self.listOfNet[self.numberOfLevels-1]
        delta_k = np.zeros((error.shape[0], error.shape[1]))

        for pattern in range(x.shape[0]):
            delta_k[pattern] = error[pattern] * self.activation[self.numberOfLevels-1](net_k, True)[pattern]
            
        listOfDelta = []
        delta_temp = delta_k

        for levels in range(self.numberOfLevels-1,0,-1):
            delta_temp = np.matmul(delta_temp,self.listOfWeightMatrices[levels][1:].T)
            
            net_j = self.listOfNet[levels-1]
            
            
            for pattern in range(x.shape[0]):
                delta_temp[pattern] = delta_temp[pattern] * self.activation[levels-1](net_j, derivative = True)[pattern]
            
            listOfDelta.append(delta_temp)
        
        # compute the gradients for each level 
        grad_output = np.matmul(delta_k.T, self.listOfHiddenRepr[self.numberOfLevels-2])

        # normalization widh batch dim
        grad_output = grad_output / x.shape[0]
        grad_hidden = []

        
        listOfDelta.reverse()

        for levels in range(0,self.numberOfLevels-1,1):
            if levels == 0:
                normGradHidden = np.matmul(listOfDelta[levels].T, np.hstack((np.ones((x.shape[0], 1)), x))).T / x.shape[0] 
                grad_hidden.append(normGradHidden)
            else:
                normGradHidden = np.matmul(listOfDelta[levels].T, self.listOfHiddenRepr[levels-1]).T / x.shape[0] 
                grad_hidden.append(normGradHidden)


        return grad_output.T, grad_hidden
    

    """
    function train, it is used to discriminate between 2 cases:
    - validationErrorCheck == True (so xValid!= None and yValid!= None) 
    It calls self.selectOptimalStartingWeights while enabling to compute validation error for each epoch
    - else (validationErrorCheck == False):
    It calls self.selectOptimalStartingWeights but without computing validation error
    """
    def train (self, X, Y, epochs=100, batch_size=None, treshold=0.1, initMode="random", numberOfRestart=5, validationErrorCheck = False, xValid = None, yValid = None):
        
        if validationErrorCheck :
            return self.selectOptimalStartingWeights(X, Y, epochs, batch_size, treshold, initMode, numberOfRestart, validationErrorCheck, xValid, yValid)
        else : 
            return self.selectOptimalStartingWeights(X, Y, epochs, batch_size, treshold, initMode, numberOfRestart)
        

    """
    function selectOptimalStartingWeights: 
    It executes multiple restarts of training with different weights initialization 
    in order to determine their best initialization
    Args:
        X: input matrix
        Y: target matrix
        epochs: numbers of Epochs
        batch_size: if != None and < Numsamples --> dimension of each batch 
                    otherwise use all training set 
        threshold: error threshold to stop training before selected number of epochs
        numberOfRestart: how many restarts it does with different weights init
        validationErrorCheck: boolean. 
            If true: 
                it computes val error for each epoch 
            Else:
                It doesn't compute val error but only training
        xValid: input matrix for val
        yValide: target matrix for val
    
    Modify:
        For the best initialization (minimum training error):
            self.initWeights is the list of initial weights matrices  
            self.optimalListOfWeightMatrices is the list of weights matrices for each level after training

    Returns:
        e = Training error (best training error of all initializations)
        listOfLogsTR = List of logs with learning curve of the training of the best weight init
        If validationErrorCheck == True:
        optLogVL = List of logs with validation_error for each epoch, of the best weight init
    """
    def selectOptimalStartingWeights(self, X, Y, epochs, batch_size, treshold, initMode, numberOfRestart, validationErrorCheck = False, xValid = None, yValid = None):
        
        # list of weight matrices after training of the best init
        self.optimalListOfWeightMatrices = []
        # at the beginning the error is +inf
        e = float("inf")
        # i = number of restart, 0 at the beginning 
        i = 0
        # list of string with the log of validation error 
        # only if validationErrorCheck is set
        if validationErrorCheck:
            optLogVL = []
        # list of string with the log of training error 
        listOfLogsTR = []
        while i < numberOfRestart:

            if validationErrorCheck :
                etmp, logVL, logTR = self.realTraining(X, Y, epochs, batch_size, treshold, initMode, validationErrorCheck, xValid, yValid)
            else : 
                etmp, logTR = self.realTraining(X, Y, epochs, batch_size, treshold, initMode, validationErrorCheck, xValid, yValid)
            
            if etmp < e:
                
                if validationErrorCheck :
                    optLogVL = copy.deepcopy(logVL)
                
                listOfLogsTR = copy.deepcopy(logTR)
                self.optimalListOfWeightMatrices = copy.deepcopy(self.listOfWeightMatrices)
                self.initWeights = copy.deepcopy(self.tmpStartWeights)
                e = etmp         
            i += 1    

        if validationErrorCheck : 
            return e, optLogVL, listOfLogsTR
        else :
            return e, listOfLogsTR

    """
    Function update_weights
    It updates net's weigths using the learning rule derived from gradient descent 
    with momentum, regularization and variable learning rate 

    Args:
        - i : current epoch of training 
        - grad_hidden: current gradient of hidden layers
        - grad_output: current gradient of output layer
        - oldGrad_hidden: previous gradient of hidden layers
        - oldGrad_output: previous gradient of output layer
        - batch_size: numbers of samples for each batch
        - num_samples: number of pattern in the dataset
    
    Modify:
        - Updates list of weight matrices (self.listOfWeightMatrices)
    
    Return: 
        - Return the new oldGrad_hidden, oldGrad_output for next iterations

    """
    def update_weights(self, i, grad_hidden, grad_output, oldGrad_hidden, oldGrad_output, batch_size, num_samples, use_mini_batch):
       
        # if variable learning rate is enable
        if self.VariableLROption :
            if i <= self.tau :
                etas = self.learning_rate_schedule(self.eta0, self.eta_tau, self.tau, i)
            else : 
                etas = self.eta_tau
        else:
            etas = self.eta0
                            
        for j in range(0, len(grad_hidden), 1):
            
            if use_mini_batch: 
                # compute the momentum contribution for the hidden gradient update rule
                # with batch normalization  
                velocityHidden = self.alpha * oldGrad_hidden[j] + ((etas) *(batch_size/num_samples) * grad_hidden[j]) 
            else:
                # compute the momentum contribution for the hidden gradient update rule 
                velocityHidden = self.alpha * oldGrad_hidden[j] + ((etas) * grad_hidden[j]) 

            # compute penalty term for regularization
            penalty_term = self.lambda_reg * self.listOfWeightMatrices[j]
            self.listOfWeightMatrices[j] = self.listOfWeightMatrices[j] + velocityHidden  - penalty_term
            # save old momentum contribution for next iteration
            oldGrad_hidden[j] = velocityHidden
                            
        if use_mini_batch:
            # compute the momentum contribution for the output gradient update rule
            # with batch normalization 
            velocityOutput = self.alpha * oldGrad_output + ((etas) *(batch_size/num_samples) * grad_output)
        else:
            # compute the momentum contribution for the output gradient update rule
            velocityOutput = self.alpha * oldGrad_output + ((etas) * grad_output)

        # compute penalty term for regularization 
        penalty_term = self.lambda_reg * self.listOfWeightMatrices[-1]
        # list[-1] = last elem of the list = weights between hidden and output
        self.listOfWeightMatrices[-1] = self.listOfWeightMatrices[-1] + velocityOutput   - penalty_term 
                            
        # save old momentum contribution for next iteration
        oldGrad_output = velocityOutput

        return oldGrad_hidden, oldGrad_output

    """
    function RealTraining performs training/gradient descent
    If validationErrorCheck == False:
        It initializes a list of weight matrices calling function initalizeWeightMatrix
    Else:
        It uses the weight loaded in the constructor 
    """
    def realTraining(self, X, Y, epochs, batch_size, treshold, initMode, validationErrorCheck = False, xValid = None, yValid = None):
        
        # if no validationErrorCheck initialize a weight matrix
        if validationErrorCheck == False : 
            self.listOfWeightMatrices = self.initalizeWeightMatrix(initMode)
        
        # save a deep copy of initial list of weight matrices for future use
        self.tmpStartWeights = copy.deepcopy(self.listOfWeightMatrices)

        # log for training 
        logTR = []  

        # log for validation
        if validationErrorCheck : logVL = []
        
        # num of pattern in the training set 
        num_samples = X.shape[0]
        # number of epochs initially 1
        i = 1
        # trainin error, initially + inf 
        e = float("inf")

        # init old gradient for momentum 
        oldGrad_hidden = [np.zeros_like(w) for w in self.listOfWeightMatrices[:-1]]
        oldGrad_output = np.zeros_like(self.listOfWeightMatrices[-1])

        # Determine if we're doing SGD or mini-batch
        use_mini_batch = batch_size is not None and batch_size < num_samples
        
        while i <= epochs and e > treshold:
            if use_mini_batch:
                # Shuffle data for each epoch
                # generate an array containing numbers from 0 to num_samples−1, randomly ordered  
                indices = np.random.permutation(num_samples)
                # apply the random permutation to the training set 
                X_shuffled = X[indices]
                Y_shuffled = Y[indices]

                # train for each batch in the training set 
                for start_idx in range(0, num_samples, batch_size):
                    end_idx = min(start_idx + batch_size, num_samples)
                    X_batch = X_shuffled[start_idx:end_idx]
                    Y_batch = Y_shuffled[start_idx:end_idx]

                    # Perform feedforward and backpropagation on the batch
                    o = self.feedForeward(X_batch, self.listOfWeightMatrices)
                    grad_output, grad_hidden = self.backPropagate(X_batch, Y_batch, o)

                    oldGrad_hidden, oldGrad_output = self.update_weights(i, grad_hidden, grad_output, oldGrad_hidden, oldGrad_output, batch_size, num_samples, use_mini_batch)

                # if i want to track the error on VL set over epoch of training 
                if validationErrorCheck:
                    # compute NN output on validation set 
                    outVal = self.feedForeward(xValid, self.listOfWeightMatrices) 
                    
                    if self.task == 'classification':
                        eVL = self.classification_error(yValid, outVal, activation=self.activationListName[-1])
                        logVL.append(f"Epoch : {i}, Classification Error  : {eVL}\n")
                    else:
                        # for regression task
                        eVL = self.mean_squared_error_loss(yValid, outVal)
                        logVL.append(f"Epoch : {i}, MSE : {eVL}\n")
                    
                    
                # keep track of training error over the epochs of training 
                o = self.feedForeward(X, self.listOfWeightMatrices)
                e = self.mean_squared_error_loss(Y, o)

                logTR.append(f"Epoch : {i}, MSE : {e}\n")
                i += 1
                  
            else:    
                # compute model output with the feedforward
                o = self.feedForeward(X, self.listOfWeightMatrices)

                # perform backprop 
                grad_output, grad_hidden = self.backPropagate(X, Y, o)

                if validationErrorCheck:
                    
                    outVal = self.feedForeward(xValid, self.listOfWeightMatrices) 
                    
                    if self.task == 'classification':
                        eVL = self.classification_error(yValid, outVal, activation=self.activationListName[-1])
                        logVL.append(f"Epoch : {i}, Classification Error : {eVL}\n")
                    else:
                        # for regression task                
                        eVL = self.mean_squared_error_loss(yValid, outVal)
                        logVL.append(f"Epoch : {i}, MSE : {eVL}\n")
                 
                # keep track of training error 
                # o output of the net is already computed in feed-forward before back-prop
                e = self.mean_squared_error_loss(Y, o)
                                
                logTR.append(f"Epoch : {i}, MSE : {e}\n")

                oldGrad_hidden, oldGrad_output = self.update_weights(i, grad_hidden, grad_output, oldGrad_hidden, oldGrad_output, batch_size, num_samples, use_mini_batch)
                
                i += 1

        if validationErrorCheck :
            return e, logVL, logTR
        else : 
            return e, logTR

    # Correct Averaging: MSE is the mean of squared errors over all data points and all features.
    def mean_squared_error_loss(self, Y, O):
        diff = Y - O
        squared_error = np.square(diff)
        mse = np.mean(squared_error)  
        return mse
    
    """
    function classification_error is used to compute classification error 
    Args: 
        - Y is the target matrix
        - o is the matrix with the output of the net 
        - activation is a string with the name of the activation 
          function at the output level
    Returns: 
        - the classification error of the net givent the target and the output 
    """
    @staticmethod
    def classification_error(Y, o, activation=None):
        
        # number of pattern 
        num_pattern = Y.shape[0]
        # take care that the output of the net 
        # is aligned with the target format 
        if activation == "sigmoid":     
            o = (o >= 0.5).astype(int)  # Converte in 0 o 1
        elif activation == "tanh": 
            # convert in -1,1
            o[o >= 0] = 1
            o[o < 0] = -1
        elif  activation == "linear":
            # convert in -1,1
            o[o >= 0] = 1
            o[o < 0] = -1
        # Compares each row of Y and o element-wise.
        # Creates a boolean matrix where each element is True if the corresponding elements in Y and o are different, and False otherwise.
        err = (Y != o).all(axis=1).astype(int)
        # Sums up the total number of misclassifications
        # Divides by the total number of data points
        return np.sum(err)/num_pattern
    

    def learning_rate_schedule(self, eta0, eta_tau, tau, step): 
        gamma = step / tau 
        eta_s = (1 - gamma) * eta0 + gamma * eta_tau
        return eta_s
    
    def predict(self, inputX, reloaded, optimalListOfWeightMatrices):
        
        # if reloaded == true then use optimalListOfWeightMatrices retreived from disk
        if reloaded :
            return self.feedForeward(inputX, optimalListOfWeightMatrices)
        
        # else use optimalListOfWeightMatrices in model just trained
        else :
            return self.feedForeward(inputX, self.optimalListOfWeightMatrices)
    

    def predict_class(self, inputX, reloaded, activation="None", optimalListOfWeightMatrices=None):

        o = self.predict(inputX, reloaded, optimalListOfWeightMatrices)

        if activation == "tanh":
            o[o >= 0] = 1
            o[o < 0] = -1  
        elif activation == "linear":
            o[o >= 0] = 1
            o[o < 0] = -1 
        elif activation == "sigmoid":
            o = (o >= 0.5).astype(int)  # Converte in 0 o 1
        return o

    def get_list_weight_matrices(self):
        return self.listOfWeightMatrices
    
    def get_list_init_weight_matrices(self):
        return self.initWeights

    # function to retreive all parameters from model
    def getParameters(self):
        return (self.optimalListOfWeightMatrices, self.units_for_levels, self.activationListName, self.eta0, self.eta_tau, self.tau, self.lambda_reg, self.alpha)

    def getOptimalWeights(self):
        return self.optimalListOfWeightMatrices

    # function to store all parameters on disk
    @staticmethod
    def saveModel(mymodelparameters, pathFile="./finalModel.txt"):
        

        with open(pathFile,  mode = 'wb') as fileModel: 
            
            modelAsString = pickle.dumps(mymodelparameters)
            fileModel.write(modelAsString)


    # function to read parameteres from disk
    @staticmethod
    def realoadModel(pathFile="./finalModel.txt"):

        with open(pathFile,  mode = 'rb') as fileModel:

            modelAsString = fileModel.read()
            mymodelparameters = pickle.loads(modelAsString)

        return mymodelparameters
   

