import pandas as pd 
from model import NeuralNetwork
import numpy as np
import myModelParameters as mmp
import os 
import matplotlib.pyplot as plt



# Carica il file 
file_path = "./dataset/ML-CUP24-TR.csv"
data = pd.read_csv(file_path, comment="#", header=None)

# drop first column not useful 
data = data.drop(data.columns[0], axis=1)

#shuffle data-set
# random_state=1 meaning that for all run the result is the same
data = data.sample(frac=1, random_state=1)

# take row number 
numsample = data.shape[0]

#select 80 % of rows
trainingPercentage = numsample * 0.8


# take the rows for k-fold validation 
selectionSet = data.iloc[:int(trainingPercentage), :]

# take the row for final testing 
testSet =  data.iloc[int(trainingPercentage):, :]


# split target from input for k-fold validation
#targetForSelection = selectionSet.iloc[:, -3:] 
#trainingForSelection = selectionSet.iloc[:, :-3]


# split target from input for test
#targetTest = testSet.iloc[:, -3:] 
#inputTest = testSet.iloc[:, :-3]

# build numpy MATRIX for training and target for training
#targetForSelection = targetForSelection.to_numpy()
#trainingForSelection = trainingForSelection.to_numpy()

#like before
#targetTest = targetTest.to_numpy() 
#inputTest = inputTest.to_numpy()

selectionSet = selectionSet.to_numpy()
testSet = testSet.to_numpy()




# number of partitions
k = 4
testSplits = mmp.myModelParameters.kFoldPartition(selectionSet, k)

listOfDict = []
listOfLog = []

validationFromIperParam = {}

for numSplit, kfolSplit in enumerate(testSplits):

    print(f"k fold number: {numSplit}")
    trSetFold = kfolSplit[0]
    vlSetFold = kfolSplit[1]

    # split target from input for k-fold validation
    x_Training = trSetFold[:, :-3]
    y_Training = trSetFold[:, -3:]
        
    x_Validation = vlSetFold[:, :-3]
    y_Validation = vlSetFold[:, -3:]

    resultOptIperParam, log = mmp.myModelParameters.doGridSearch(x_Training, x_Validation, y_Training, y_Validation, [12,24,3], ['elu','linear'], task = 'regression')

    for key in resultOptIperParam.keys():
        valError = resultOptIperParam[key][1]
        trError = resultOptIperParam[key][0]
        if numSplit == 0:
            validationFromIperParam[key] = ((valError / k), (trError / k))
        else :
            validationFromIperParam[key] = (validationFromIperParam[key][0] + (valError / k), validationFromIperParam[key][1] + (trError / k))

    listOfDict.append(resultOptIperParam)
    listOfLog.append(log)



with open("k_fold_result.txt", "w") as file:
    for key in validationFromIperParam.keys():
        file.write(f"keys: {key}  (Val error: {validationFromIperParam[key][0]} ,  TR error: {validationFromIperParam[key][1]})\n")




for dict in listOfDict:
    # print(dict)
    # Itera su tutte le chiavi del dizionario
    for key, value in dict.items(): 
        # Estrai i primi due valori della tupla
        training_error = value[0]
        validation_error = value[1]
    
        # Stampa il messaggio desiderato
        print(f'Per gli iperparametri {key}: il training error è "{training_error}" e il validation error è "{validation_error}".')

print(f'*******************************************************************')
# Salva ogni log in un file separato

# Specifica la directory di output per i file
log_directory = "./log"

# Crea la directory se non esiste
os.makedirs(log_directory, exist_ok=True)

# Salva ogni log in un file separato
for index, log in enumerate(listOfLog, start=1):
    file_name = f"log_{index}.txt"  # Nome del file
    file_path = os.path.join(log_directory, file_name)  # Percorso completo

    # Converte il contenuto del log in stringa
    if isinstance(log, list):
        log_content = "\n".join(log)  # Unisci i contenuti della lista in una stringa con newline
    else:
        log_content = str(log)  # Converte in stringa se non è già

    # Scrive nel file
    with open(file_path, "w") as file:
        file.write(log_content)  # Scrive il contenuto del log nel file
        print(f"Salvato: {file_path}")



