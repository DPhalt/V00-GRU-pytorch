from pandas_datareader import data
import matplotlib.pyplot as plt
import time
import numpy as np
import torch
from GRU import GRU
from sklearn.preprocessing import MinMaxScaler



start_date = '2010-01-01'  #The starting date
end_date = '2020-12-31'    # The ending date

panel_data = data.DataReader('VOO', 'yahoo', start_date, end_date) #Reads Vanguard S&P500 data from Yahoo Finance into a dataframe

col = panel_data[['Adj Close']] #isolates the column that I intend on using as input


scaler = MinMaxScaler(feature_range=(-1,1))
col['Adj Close'] = scaler.fit_transform(col['Adj Close'].values.reshape(-1,1))  #Normalises the data by squeezing it between -1 and 1, ref [Kar21] in report



def makeBatch(stock, batchsize):
    raw = stock.to_numpy() #converts to numpy ndarray

    batched = [] #prepares an array that will store each of the batches

    for i in range(len(raw) - batchsize): #This is calculated in order to prevent an index-out-of-bounds error
        batched.append(raw[i: i + batchsize]) #raw[i: i + batchsize] proceeds to seperate the data into batches of ndarrays with each loop, each batch being one time-step ahead of its predecessor
                                            #  This allows for consistency, which in turn makes it easier for the GRU to see which information to keep and which to forget
                                            # Batched becomes an array of ndarrays

    data = np.array(batched)  # Since batched is a normal array, not a numpy ndarray, this allows for batched to be converted into an ndarray for its properties.

    test_size = int(np.round(0.20 * data.shape[0]))  #measures the size of the amount that's going to be sliced off for the test data. In this case, it is 1/5 of the data rounded up.
    train_size = data.shape[0] - test_size          #this uses the testing size to calculate the size of the training data, as the test_size was rounded up, so to avoid clashes in the calculations

    x_train = data[:train_size, : -1]     #Takes all the arrays up until the max training size and,of those arrays, stores every input except the last. Remember, each batch is one time-step ahead of its previous batch, so essentially this takes all the data needed to produce an output but without taking the final output itself.
    y_train = data[:train_size, -1] #This takes the final output mentioned in the previous comment. This takes the same amount of arrays as x_train. However, it stores the final amount of each of those arrays stored in x_train.

    x_test = data[train_size:, : -1] #The same as x_train. Only it stores the last section of arrays, past the max training size.
    y_test = data[train_size:, -1] #The same as y_train. Only it stores the last section of arrays, past the max training size.

    return [x_train, y_train, x_test, y_test]


x_train, y_train, x_test, y_test = makeBatch(col, 30) #calls the above function to seperate the data into its batches


x_train_tensor = torch.from_numpy(x_train).type(torch.Tensor)
y_train_tensor = torch.from_numpy(y_train).type(torch.Tensor)
x_test_tensor = torch.from_numpy(x_test).type(torch.Tensor)     #converting the np.ndarrays to torch.tensors to run through the GRU
y_test_tensor = torch.from_numpy(y_test).type(torch.Tensor)





epochs = 180
histGRU = np.zeros(epochs) #preps a list of zeroes of the epoch length in order to document how the loss is minimised


input_num = 1
hidden_num = 120            #These are the variables being passed as parameters to the GRU network.
n_layers = 1
output_num = 1


modelGRU = GRU(input_num, hidden_num, output_num, n_layers)     # This is where the GRU is initialised with the variables stated above
criterion = torch.nn.MSELoss(reduction= 'mean')                 # This is our loss function. reduction = 'mean' is my being redundant in order to be sure the average is taken.
optimiserGRU = torch.optim.Adam(modelGRU.parameters(), lr=0.02) # Adam is used as an optimiser as it is faster than Stochastic Gradient Descent
start_time = time.time() #Starts a timer to test how fast the network trains
def trainer(GRU, optim, crit, histiGRU): #This is the function that trains the network, It takes in the network, criterion, optimiser and histGRU to record the loss minimization
    for i in range(epochs):
        y_pred = GRU(x_train_tensor) #Here is when the network applies the feedforward algorithm and makes a prediction.
        loss = crit(y_pred, y_train_tensor) #This calculates how far off the prediction is from the actual answer.

        if i % 5 == 0:
            print("Epoch ", i, "MSE GRU: ", loss.item())

        histiGRU[i] = loss.item() #This adds the current loss value to the histGRU array.

        optim.zero_grad() #Zeroes out the gradients, else they accumulate and cause the gradients to not point to the min/max of the loss

        loss.backward() #backpropagates and differentiates the entire graph with respect to the loss

        optim.step() #updates the weights and biases

    return loss, y_pred, histiGRU


lossGRU, y_pred_GRU, histGRU = trainer(modelGRU, optimiserGRU, criterion, histGRU)
training_time = time.time() - start_time #Acquires the total time in which the Network trained

print("Training time: " + "{}".format(training_time))
plt.plot(histGRU, label=" Final GRU Training loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")      #Plots and shows HistGRU
plt.legend()
plt.show()

checkpoint = {"input_num": input_num,
              "hidden_num": hidden_num,
              "output_num" : output_num,
              "num_layers": n_layers,
              "state_dict": modelGRU.state_dict()       #Stores the model's details in a dictionary in order to save it
            }
torch.save(checkpoint, "VOO_GRU.pth") #Saves the model to an external .pth file

# start of eval
modelGRU.eval() #Switches the model's focus from training to prediction.

with torch.no_grad(): #Sets all the requires_grad to false, so no gradients are being changed
    out_data = modelGRU(x_test_tensor) #The model makes its evaluation


plt.plot(y_test, label="Original")
plt.plot(out_data, label="Evaluation") #Plots the model's evaluation and the original output side-by-side in order to test the model's accuracy


plt.legend()
plt.show()

y_test_extended = np.append(y_train, y_test)
out_data_extended = np.append(y_train, out_data)
plt.plot(col.index[30:], y_test_extended, label="Original")
plt.plot(col.index[30:], out_data_extended, label="Evaluation") #Appends the training data so as to show how the graph looks originally, as well as with the prediction.


plt.legend()
plt.show()
