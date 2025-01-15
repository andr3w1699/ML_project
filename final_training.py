# import libraries 
import pandas as pd 
from model import NeuralNetwork
import numpy as np
import myModelParameters as mmp
import os 
import matplotlib.pyplot as plt


# Carica il file del dataset di training
file_path = "./dataset/ML-CUP24-TR.csv"
data_training = pd.read_csv(file_path, comment="#", header=None)

# Carica il file del dataset di test 
file_path_test = "./dataset/ML-CUP24-TS.csv"
data_blind_test = pd.read_csv(file_path_test, comment="#", header=None)

#print(data_blind_test)

# drop first column not useful because ID column
data_training = data_training.drop(data_training.columns[0], axis=1)

#shuffle data-set
# random_state=1 meaning that for all run the result is the same
data_training = data_training.sample(frac=1, random_state=1)

# take row number 
numsample = data_training.shape[0]

#select 80 % of rows
trainingPercentage = numsample * 0.8


# take the rows for k-fold validation 
selectionSet = data_training.iloc[:int(trainingPercentage), :]

# take the row for final testing 
testSet =  data_training.iloc[int(trainingPercentage):, :]


Y_selection = selectionSet.iloc[:, -3:] 
Y_selection = Y_selection.to_numpy()
X_selection = selectionSet.iloc[:, :-3]
X_selection = X_selection.to_numpy()

Y_Test = testSet.iloc[:, -3:] 
Y_Test = Y_Test.to_numpy()
X_Test = testSet.iloc[:, :-3]
X_Test = X_Test.to_numpy()


# retraining on selection set after model selection 

# create model parameter
prm =  mmp.myModelParameters(None, [12,24,3], ['elu','linear'], True, 0.01, 0.01, 1000 , 0.0001, 0.5, True , "regression")
# create model 
model = NeuralNetwork(prm)
# train the model - For final retraining we have no validation 
trainError, logVL, LogsTR = model.train(X_selection, Y_selection, 1000, None, 0.0001, "xavier", 20, True, X_selection, Y_selection)
# make the prediction on training
prediction_on_training = model.predict(X_selection, False, None)
# make the prediction on validation
prediction_on_Test = model.predict(X_Test, False, None)

# RISK ASSESMENT ON INTERNAL TEST 

# compute MSE and MEE on training and Test 
MSE_training = model.mean_squared_error_loss(Y_selection, prediction_on_training)
MSE_test = model.mean_squared_error_loss(Y_Test, prediction_on_Test)
MEE_training = model.mean_euclidean_error_loss(Y_selection, prediction_on_training)
MEE_test = model.mean_euclidean_error_loss(Y_Test, prediction_on_Test)


print(F"MSE on training(Selection Set): {MSE_training} vs MSE on Internal Test: {MSE_test}\nMEE on training(Selection Set): {MEE_training} vs MEE on Internal Test: {MEE_test}")

print(F"RISK ASSESMENT:\nEstimated Error on internal test set, MSE: {MSE_test}, MEE:{MEE_test}")

"""
xasses = []
yasses = []
xassesVL = []
yassesVL = []
        
for str in logVL:
    Mytuple = str.split(",")
            
    xassesVL.append(int(Mytuple[0].split(":")[1]))
    yassesVL.append(float(Mytuple[1].split(":")[1]))
        
        
for str in LogsTR:
    Mytuple = str.split(",")
            
    xasses.append(int(Mytuple[0].split(":")[1]))
    yasses.append(float(Mytuple[1].split(":")[1]))

plt.plot(np.array(xasses), np.array(yasses), linestyle="dashed", label="MEE on TR")
#plt.plot(np.array(xassesVL), np.array(yassesVL),label="MEE on TS")        
#plt.plot(np.array(xasses), np.array(yasses), linestyle="dashed", label="MSE on TR")
#plt.plot(np.array(xassesVL), np.array(yassesVL),label="MSE on VL")
plt.legend()
        
plt.title("Learning curve retraining on Selection Set")
#plt.title("MSE TR vs MSE Validation")
plt.xlabel("epochs")
plt.ylabel("Mean Euclidean Error (MEE)")
#plt.ylabel("Mean Squared Error (MSE)")
plt.grid(True)
plt.show()

"""

# take the entire dataset for final retraining after risk assesment 

Y_training = data_training.iloc[:, -3:].to_numpy()
X_training = data_training.iloc[:, :-3].to_numpy()

# create model - use the same hyper param since the model 
model_final = NeuralNetwork(prm)
# train the model - For final retraining we have no validation 
trainError, logVL, LogsTR = model_final.train(X_training, Y_training, 1000, None, 0.0001, "xavier", 20, True, X_training, Y_training)
# make the prediction on training
prediction_on_final_training = model_final.predict(X_training, False, None)
MSE_training_final = model_final.mean_squared_error_loss(Y_training, prediction_on_final_training)
MEE_training_final = model_final.mean_euclidean_error_loss(Y_training, prediction_on_final_training)

print(f"Final retraining:\nMSE: {MSE_training_final}, MEE:{MEE_training_final}")


xasses = []
yasses = []
xassesVL = []
yassesVL = []
        
for str in logVL:
    Mytuple = str.split(",")
            
    xassesVL.append(int(Mytuple[0].split(":")[1]))
    yassesVL.append(float(Mytuple[1].split(":")[1]))
        
        
for str in LogsTR:
    Mytuple = str.split(",")
            
    xasses.append(int(Mytuple[0].split(":")[1]))
    yasses.append(float(Mytuple[1].split(":")[1]))

plt.plot(np.array(xasses), np.array(yasses), linestyle="dashed", label="MEE on TR")
#plt.plot(np.array(xassesVL), np.array(yassesVL),label="MEE on TS")        
#plt.plot(np.array(xasses), np.array(yasses), linestyle="dashed", label="MSE on TR")
#plt.plot(np.array(xassesVL), np.array(yassesVL),label="MSE on VL")
plt.legend()
        
plt.title("Learning curve final retraining")
#plt.title("MSE TR vs MSE Validation")
plt.xlabel("epochs")
plt.ylabel("Mean Euclidean Error (MEE)")
#plt.ylabel("Mean Squared Error (MSE)")
plt.grid(True)
plt.show()


