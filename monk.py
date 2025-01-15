import pandas as pd 
from model import NeuralNetwork
import numpy as np
import myModelParameters as mmp
import os 
import matplotlib.pyplot as plt

import math 



modelParametersFile = "./finalModel.txt"




# Carica il file 
file_path = "./dataset/monk/monks-3.train"
data = pd.read_csv(file_path, sep=" ", header=None, skipinitialspace=True)

path_test = "./dataset/monk/monks-3.test"
test_set = pd.read_csv(path_test, sep=" ", header=None, skipinitialspace=True)

# drop last column not useful 
data = data.drop(data.columns[-1], axis=1)

data_test = test_set.drop(test_set.columns[-1], axis=1)

target = data.iloc[:, 0]  # Seleziona la prima colonna (indice 0)
target_test = data_test.iloc[:, 0]

training_set = data.iloc[:, 1:]  # Tutte le colonne tranne la prima
training_set = pd.get_dummies(training_set, columns=training_set.columns[0:], drop_first=False)
input_test = data_test.iloc[:, 1:]
input_test = pd.get_dummies(input_test, columns=input_test.columns[0:], drop_first=False)

# convert to numpy 
x = training_set.to_numpy()
x_test = input_test.to_numpy()

#print(x_test)
#print(x_test.shape[0])
#print(x_test.shape[1])
#print("\n")

y = target.to_numpy().reshape(-1, 1)
y_test = target_test.to_numpy().reshape(-1, 1)

# if you want to use tanh 
y[y == 0] = -1
y_test[y_test == 0] = -1

#print(y)
#print(y.shape[0])
#print(y.shape[1])

# added selection mode (standard mode, debug mode)
debugMode = False

while True:
    try :
        # inp = input("insert \"y\" for debug mode or others for standard mode\n")
        inp = "y"
        if inp == "y":
            debugMode = True
        else: debugMode = False
        break
    except ValueError :
        print(f"errore: {ValueError}")

if debugMode == False and os.path.isfile(modelParametersFile) :

    objectReloaded = NeuralNetwork.realoadModel()
    model = NeuralNetwork(objectReloaded, debugMode)
    
    result = model.predict_class(x_test, True, "tanh", objectReloaded.weights)
    print(f"classification error on TR set: {model.classification_error(result, y_test)}")

else :


    #(xTrain, xValid, yTrain, yValid, units_for_levels, activation, debugMode)
    # inputG = input("insert \"g\" for Grid Search, otherwise another character\n")
    inputG = "g"
    # do grid search 
    if inputG == "g":
        
        #  optModel, resultOptIperParam, optimalKeys, optimalValue, LogsTR, logVL

        modelWithGridSearch, result, optimalKeys, optimalValue, optLogsTR, logVL = mmp.myModelParameters.doGridSearch(x, x_test, y, y_test, [17,4,1], ['sigmoid', 'linear'], task = 'classification')
        xasses = []
        yasses = []
        xassesVL = []
        yassesVL = []
        
        for str in logVL:
            Mytuple = str.split(",")
            
            xassesVL.append(int(Mytuple[0].split(":")[1]))
            yassesVL.append(float(Mytuple[1].split(":")[1]))
        
        
        for str in optLogsTR:
            Mytuple = str.split(",")
            
            xasses.append(int(Mytuple[0].split(":")[1]))
            yasses.append(float(Mytuple[1].split(":")[1]))
        
        plt.plot(np.array(xasses), np.array(yasses), linestyle="dashed",  label="Accuracy TR")
        plt.plot(np.array(xassesVL), np.array(yassesVL), label="Accuracy TS")
        plt.legend()
        

        #plt.title("MEE TR vs MEE Validation")
        #plt.title("MSE TR vs MSE TEST")
        plt.title("Accuracy TR vs Accuracy TEST")
        plt.xlabel("epochs")
        # plt.ylabel("Mean Euclidean Error (MEE)")
        # plt.ylabel("Mean Squared Error (MSE)")
        plt.ylabel("Classification Accuracy")
        plt.grid(True)
        plt.show()
        
    else :    
        # instantiate the neural network  units_for_levels, activation, VariableLROption = False, eta0=0.8, eta_tau=0.5, tau=100, lambda_reg=0.01, alpha = 0.9
        param = mmp.myModelParameters(None, [17,4,1], ['sigmoid','sigmoid'], True, 0.8, 0.5, 100, 0.01, 0.9)
        model = NeuralNetwork(param, debugMode)

        model.train(x, y, 20, 0.00001, "xavier", 1)

        # retreive all parameters from model
        mymodelparameters = mmp.myModelParameters(*model.getParameters())

        # save all parameters in a file
        model.saveModel(mymodelparameters)

        # predict after training
        result = model.predict_class(x_test, False)

        print(f"classification error on TR set: {model.classification_error(result, y_test)}")

        """
        result = model.predict(np.array([ [4],
                            [8],
                            [16],
                            [32],
                            [64],
                            ]) )
        """

        # print(f" predizione del log in base 2 di 1, 2 , 4 , 8 e 16  : {result}")

    