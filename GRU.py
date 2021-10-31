import torch
import torch.nn as nn


class GRU(nn.Module):
    def __init__(self, input_num, hidden_num, output_num, n_layers, drop_prob = 0.8):  #This is what constructs the GRU. The details of it are covered in the report
        super(GRU, self).__init__() #Allows for this class to use methods from its parent class
        self.hidden_num = hidden_num
        self.n_layers = n_layers            #Stores the current hidden_num and n_layers in this instance of the GRU

        self.gru = nn.GRU(input_num, hidden_num, batch_first= True, dropout= drop_prob) #This module encompasses all the equations and gates listed in the report, and structures it appropriately.
        self.fc = nn.Linear(hidden_num, output_num) #This is the fully connected layer that applies a linear transformation of "y = W^T * X + b" on the data



    def forward(self, x): #This is what is used during the feedforward process of the algorithm
        h0 = torch.zeros(self.n_layers, x.size(0), self.hidden_num).requires_grad_() #initialises the hidden layer
        out, (hn) = self.gru(x, (h0.detach())) #Returns the output as well as the operated-on hidden state for the next loop
        out = self.fc(out[:, -1, :]) #passes the output, which has been sliced in such a way that it returns the final value of the hidden state
        return out

