from model import NeuralNetwork
from sklearn.model_selection import KFold

class myModelParameters:
    """
    It includes all the parameters to initialize the NeuralNetwork
    
    Params:
        - weights: list of weight matrices. If validationErrorCheck == True, it reloads list of weights of the NN  from passed weights
        - units_for_levels: list of units for each level. Number of leves is equal to [Len of the list - 1]
        - activation: list of strings with activation functions
        - VariableLROption: boolean to set variable learning rate 
                if step <= tau:
                    gamma = step / tau 
                    eta_s = (1 - gamma) * eta0 + gamma * eta_tau
                else:
                    eta_s = eta_tau
        - eta0: initial eta in variable learning rate
        - eta_tau: final eta in variable learning rate
        - tau: threshold of iteration after which I have to use fixed learning rate
        - lambda_reg: regularization coeff.
        - alpha: momentumm coeff.
        - validationErrorCheck: boolean. If validationErrorCheck == True, it reloads list of weights of the NN from passed weights and computes the error on validation set for each epoch.
    """



    def __init__(self, weights, units_for_levels, activation, VariableLROption = False, eta0=0.8, eta_tau=0.5, tau=100, lambda_reg=0.01, alpha = 0.9, validationErrorCheck = False, task = None):
        
        self.validationErrorCheck = validationErrorCheck
        self.VariableLROption = VariableLROption
        self.weights = weights
        self.units_for_levels = units_for_levels
        self.activation = activation
        self.eta0 = eta0
        self.eta_tau = eta_tau
        self.tau = tau
        self.lambda_reg = lambda_reg
        self.alpha = alpha
        self.task = task
    



    # function to split data set in k-fold
    @staticmethod
    def kFoldPartition(trainingForSelection, k):


        kf = KFold(n_splits=k, shuffle=False)
        splits = []
        for trainIDX, validIDX in kf.split(trainingForSelection):
            splits.append((trainingForSelection[trainIDX] , trainingForSelection[validIDX]))    
        
        return splits 


    @staticmethod
    def doGridSearch(xTrain, xValid, yTrain, yValid, units_for_levels, activation, task):

        resultOptIperParam = {}

        optimalKeys = (None, None, None, None)
        optimalValue = (None, float("inf"), None)

        """
        rangeEta = [0.01, 0.03, 0.06, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        rangeLambda = [0.0001, 0.001, 0.01, 0.1, 0,2]
        rangeAlpha = [0.01, 0.03, 0.06, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        rangeEpochs = [500, 600, 700, 800, 900, 1000]
        """ 
        """
        rangeEta = [0.8, 0.9]
        rangeLambda = [0.0001, 0.001]
        rangeAlpha = [0.8, 0.9]
        rangeEpochs = [10, 20]
        """
        
        rangeEta0 = [0.001, 0.01]
        rangeLambda = [0.0001, 0.001]
        rangeAlpha = [0.5, 0.1]
        rangeEpochs = [1000]
        rangeEtaFinal = [0.001, 0.01]
        rangeTau = [1000]
        rangeinitModes = ["xavier", "random"]
        rangeRandomRestarts = [10]
        rangeMiniBatch = [None, 1, 32, 64]
        
        """
        rangeEta0 = [0.8]
        rangeLambda = [0.001]
        rangeAlpha = [0.9]
        rangeEpochs = [500]
        rangeEtaFinal = [0.8]
        """

        # instantiate the neural network  units_for_levels, activation, VariableLROption = False, eta0=0.8, eta_tau=0.5, tau=100, lambda_reg=0.01, alpha = 0.9


        optLogsTR = []
        startWeightsForOptimalTraining = []

        optModel = None
        # start with learning rate
        for idxEta0, eta0 in enumerate(rangeEta0):
            for idxetaF, etaFinal in enumerate(rangeEtaFinal):
                for idxLambda , Lambda in enumerate(rangeLambda):
                    for idxAlpha, Alpha in enumerate(rangeAlpha):
                        for idxepochs, epochs in enumerate(rangeEpochs):
                            for idxMiniBatch, miniBatch in enumerate(rangeMiniBatch):
                                for idxRandomRestart, randomRestart in enumerate(rangeRandomRestarts):
                                    for idxTau, tau in enumerate(rangeTau) :
                                        for idxInitMode, mode in enumerate(rangeinitModes) : 

                                            print(f"idxs combination: idxMiniBatch : {idxMiniBatch}, idxRandomRestart: {idxRandomRestart}, idxTau : {idxTau} , idxInitMode : {idxInitMode},  idxEta0 : {idxEta0} , idxetaF : {idxetaF} , idxLambda : {idxLambda} , idxAlpha : {idxAlpha} , idxepochs : {idxepochs}")                                            
                                            prm = myModelParameters(None, units_for_levels, activation, True, eta0, etaFinal, tau , Lambda, Alpha, False , task)
                                            model = NeuralNetwork(prm)
                                            

                                            trainError, LogsTR = model.train(xTrain, yTrain, epochs, miniBatch , 0.0001, mode , randomRestart, False, None, None)
                                            
                                            if task == 'classification':
                                                #for classification 
                                                result = model.predict_class(xValid, False, activation[-1], None )
                                                valError = model.classification_error(yValid, result, activation[-1])
                                            else:
                                                #for regression
                                                result = model.predict(xValid, False, None)
                                                valError = model.mean_squared_error_loss(yValid, result)
                                                
                                                
                                            optWeights = model.getOptimalWeights()
                                            resultOptIperParam[(eta0, etaFinal, Lambda, Alpha, epochs, mode, tau, randomRestart, miniBatch)] = (trainError, valError, optWeights, model.get_list_init_weight_matrices())
                                            if valError < optimalValue[1] :
                                                optimalValue = (trainError, valError, optWeights)
                                                optimalKeys = (eta0, etaFinal, Lambda, Alpha, epochs)
                                                optModel = model
                                                optLogsTR = LogsTR
                                                startWeightsForOptimalTraining = optModel.get_list_init_weight_matrices()


                            

        """


        # retraining model with best hiperparameters
        # weights, units_for_levels, activation, VariableLROption = False, eta0=0.8, eta_tau=0.5, tau=100, lambda_reg=0.01, alpha = 0.9, validationErrorCheck = False, task = None
        modelToBuildValidationError = NeuralNetwork(myModelParameters(startWeightsForOptimalTraining, units_for_levels, activation, True, optimalKeys[0], optimalKeys[1], 100, optimalKeys[2], optimalKeys[3], True, task = 'classification'))
        trainError, logVL, LogsTR = modelToBuildValidationError.train(xTrain, yTrain, optimalKeys[4], None, 0.0001, "random", 1, True, xValid, yValid)

        result = modelToBuildValidationError.predict_class(xValid, False, activation[-1], None)
        valError = modelToBuildValidationError.classification_error(yValid, result, activation[-1])
        
        print(f"****************************************************************\n")
        print(f"****************************************************************\n")
        print(f"valError :\n")
        print(f"{valError}\n")
        


        print(f"****************************************************************\n")
        print(f"****************************************************************\n")


        print(f"logVL:\n")
        for kk in logVL : 
            print(f" {kk}")
        
        print(f"****************************************************************\n")
        print(f"****************************************************************\n")
        print(f"****************************************************************\n")
        print(f"****************************************************************\n")
        print(f"****************************************************************\n")

        print(f"LogsTR:\n")
        for kk in LogsTR : 
            print(f" {kk}")



        return optModel, resultOptIperParam, optimalKeys, optimalValue, LogsTR, logVL
        """

        return resultOptIperParam, optLogsTR