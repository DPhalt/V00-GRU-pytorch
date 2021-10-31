# V00-GRU-pytorch
This is an implementation of the Gated Recurrent Unit in PyTorch with regards to the prediction of the Vanguard S&P500 index fund.

Note that you can change the stock that you want to analyse by finding the appropriate abbrieviation. For the Vanguard S&P500, the abbrieviation is VOO. 
Hence the name of this repository


The GRU.py file houses the Gated Recurrent Unit class.


The TrainGRU.py file houses the steps taken in order to train the network.


The VOO_GRU.pth file houses the trained model's state_dict (basically the blueprint used to turn an newly initialised GRU into the VOO_GRU), as well as the network's input size, hidden layer size, output size and the number of layers used in the trained model.
