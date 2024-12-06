from model import NeuralNetwork
import numpy as np
import math 



# added selection mode (standard mode, debug mode)
debugMode = False

while True:
    try :

        input = input("insert \"true\" for debug mode. false otherwise\n")
        if input == "true":
            debugMode = True
        else: debugMode = False
        break
    except ValueError :
        print(f"errore: {ValueError}")

if debugMode == False:
    print(f"Training...\n")
# instantiate the neural network
model = NeuralNetwork([1,4,2,1], ['sigmoid','sigmoid','linear'], debugMode)


"""
matrix = np.array([ [1],
                    [2],
                    [3],
                    [4],
                    [5],
                     ]) 
#output = model.predict(matrix)
# print("Stampa finale")
# print(output)
"""
x = np.random.uniform(1,20,200).reshape(-1,1)
y = np.log2(x) #math.log(x, 2)

# model.train(matrix, np.array([[2], [4], [6], [8], [10]]))
model.train(x, y, 1000, 0.0001)

result = model.predict(np.array([ [1],
                    [2],
                    [4],
                    [8],
                    [16],
                     ]) )


print(f" predizione del log in base 2 di 1, 2 , 4 , 8 e 16  : {result}")
