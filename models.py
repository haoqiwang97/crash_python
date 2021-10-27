from data import *
import torch
import torch.nn as nn
from torch import optim
import random


class TrivialNN(nn.Module):
    def __init__(self, inp, hid, out):
        """
        Construct the computation graph

        Parameters
        ----------
        inp : int
            size of input
        hid : int
            size of hidder layer
        out : int
            size of output

        Returns
        -------
        None.

        """
        super(TrivialNN, self).__init__()
        
        self.fc1 = nn.Linear(inp, hid)
        self.l1 = nn.ReLU()
        self.fc2 = nn.Linear(hid, hid)
        self.l2 = nn.ReLU()
        self.fc3 = nn.Linear(hid, out)
        
    def forward(self, x):
        out = self.fc1(x)
        out = self.l1(out)
        out = self.fc2(out)
        out = self.l2(out)
        out = self.fc3(out)
        return out
        
    def predict(self, x):
        with torch.no_grad():
            return self.forward(x)
            

def train_trivialnn(args, X_train, y_train, X_val, y_val):
    X_train = torch.from_numpy(np.array(X_train)).float()
    y_train = torch.from_numpy(np.array(y_train)).float().view((-1, 1))
    
    inp_size = X_train.shape[1]
    hid_size = args.hidden_size
    out_size = 1
    
    n_epochs = args.num_epochs
    batch_size = args.batch_size
    n_batches = int(len(X_train) / batch_size)
    lr = args.lr
    
    device = args.device
    trivialnn = TrivialNN(inp_size, hid_size, out_size)
    trivialnn.to(device)
    
    optimizer = optim.Adam(trivialnn.parameters(), lr=lr)
    loss_func = nn.MSELoss()
    
    for epoch in range(0, n_epochs):
        ex_indices = [i for i in range(0, len(X_train))]
        random.shuffle(ex_indices)
        total_loss = 0.0
        
        for batch_idx in range(n_batches):
            # print("batch_idx", batch_idx)
            batch_ex_indices = ex_indices[batch_idx * batch_size: (batch_idx + 1) * batch_size]
            batch_loss = 0.0
            
        # for idx in batch_ex_indices:
        # for idx in ex_indices:
            X = X_train[batch_ex_indices].to(device)
            y = y_train[batch_ex_indices].to(device)
            
            trivialnn.zero_grad()
            prediction = trivialnn.forward(X)
            loss = loss_func(prediction, y)
            
            # batch_loss += loss
            total_loss += loss
            
        # batch_loss /= batch_size
        # batch_loss.backward() # compute gradients
            loss.backward()
            optimizer.step() # apply gradients
        print("Total loss on epoch %i: %f" % (epoch, total_loss))
        
    return trivialnn
                