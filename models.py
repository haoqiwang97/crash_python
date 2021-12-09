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


class SeveritySumNN(nn.Module):
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
        super(SeveritySumNN, self).__init__()
        
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
        

def train_severity_sum_nn(args, X_train, y_train, X_val, y_val):
    X_train = torch.from_numpy(np.array(X_train)).float()
    y_train = torch.from_numpy(np.array(y_train)).float()
    
    inp_size = X_train.shape[1]
    hid_size = args.hidden_size
    out_size = y_train.shape[1]
    
    n_epochs = args.num_epochs
    batch_size = args.batch_size
    n_batches = int(len(X_train) / batch_size)
    lr = args.lr
    
    device = args.device
    severity_sum_nn = SeveritySumNN(inp_size, hid_size, out_size)
    severity_sum_nn.to(device)
    
    optimizer = optim.Adam(severity_sum_nn.parameters(), lr=lr)
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
            
            severity_sum_nn.zero_grad()
            prediction = severity_sum_nn.forward(X)
            loss = loss_func(prediction, y)
            
            # batch_loss += loss
            total_loss += loss
            
        # batch_loss /= batch_size
        # batch_loss.backward() # compute gradients
            loss.backward()
            optimizer.step() # apply gradients
        print("Total loss on epoch %i: %f" % (epoch, total_loss))
        
    return severity_sum_nn


class SeverityIndNN(nn.Module):
    def __init__(self, inp, hid, out, n_geo_feats):
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
        super(SeverityIndNN, self).__init__()
    
        self.gru = nn.GRU(input_size=inp,
                          hidden_size=hid,
                          num_layers=1,
                          batch_first=False,
                          dropout=0)
        self.fc1 = nn.Linear(hid, n_geo_feats)
        self.l1 = nn.ReLU()
        self.fc2 = nn.Linear(n_geo_feats, hid)
        
        self.l2 = nn.ReLU()
        self.fc3 = nn.Linear(hid, out)
        
    def forward(self, x, geo_feats):
        output, h_n = self.gru(x.float().unsqueeze(1))
        fc1_out = self.fc1(output)
        l1_out = self.l1(fc1_out)
        #   fc_out = self.fc(output[-1, -1, :]).view(-1)
        geo_out = self.fc2(geo_feats)
        
        # self.fc3(self.l2(geo_out))
        # add relu, fc

        return self.fc3(self.l2(geo_out))# geo_out
        
    def predict(self, x, geo_feats):
        with torch.no_grad():
            return self.forward(x, geo_feats)
        

def train_severity_ind_nn(args, train_exs, test_exs):
    #X_train = torch.from_numpy(np.array(X_train)).float()
    #y_train = torch.from_numpy(np.array(y_train)).float()
    
    
    inp_size = train_exs[0].x_temporal.shape[1]
    hid_size = args.hidden_size
    out_size = inp_size
    geo_size = train_exs[0].x_geo.shape[0]
    
    n_epochs = args.num_epochs
    batch_size = 10#args.batch_size
    n_batches = int(len(train_exs) / batch_size)
    lr = args.lr
    
    device = args.device
    severity_ind_nn = SeverityIndNN(inp_size, hid_size, out_size, geo_size)
    severity_ind_nn.to(device)
    
    optimizer = optim.Adam(severity_ind_nn.parameters(), lr=lr)
    loss_func = nn.MSELoss()
    
    for epoch in range(0, n_epochs):
        ex_indices = [i for i in range(0, len(train_exs))]
        random.shuffle(ex_indices)
        total_loss = 0.0
        
        for batch_idx in range(n_batches):
            # print("batch_idx", batch_idx)
            batch_ex_indices = ex_indices[batch_idx * batch_size: (batch_idx + 1) * batch_size]
            batch_loss = 0.0
            
            for idx in batch_ex_indices:
        # for idx in ex_indices:
                X = torch.from_numpy(train_exs[idx].x_temporal).float().to(device)
                geo = torch.from_numpy(train_exs[idx].x_geo).float().to(device)
                y = torch.from_numpy(train_exs[idx].y).float().to(device)
            
                severity_ind_nn.zero_grad()
                prediction = severity_ind_nn.forward(X, geo)
                loss = loss_func(prediction, y)
            
                batch_loss += loss
                total_loss += loss
            
        # batch_loss /= batch_size
            batch_loss.backward() # compute gradients
            #batch_loss.backward()
            optimizer.step() # apply gradients
        print("Total loss on epoch %i: %f" % (epoch, total_loss))
        
    return severity_ind_nn


class SeverityIndNN2(nn.Module):
    def __init__(self, inp, hid, out, n_geo_feats):
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
        super(SeverityIndNN2, self).__init__()
    
        self.gru = nn.GRU(input_size=inp,
                          hidden_size=hid,
                          num_layers=1,
                          batch_first=False,
                          dropout=0)
        self.fc1 = nn.Linear(n_geo_feats + hid, hid)
        self.l1 = nn.ReLU()
        self.fc2 = nn.Linear(hid, hid)
        
        self.l2 = nn.ReLU()
        self.fc3 = nn.Linear(hid, out)
        
    def forward(self, x, geo_feats):
        output, h_n = self.gru(x.float().unsqueeze(1))
        
        
        merged = torch.cat((h_n.squeeze(), geo_feats))
        
        fc1_out = self.fc1(merged)
        
        l1_out = self.l1(fc1_out)
        #   fc_out = self.fc(output[-1, -1, :]).view(-1)
        fc2_out = self.fc2(l1_out)

        return self.fc3(self.l2(fc2_out))# geo_out
        
    def predict(self, x, geo_feats):
        with torch.no_grad():
            return self.forward(x, geo_feats)
        

def train_severity_ind_nn2(args, train_exs, test_exs):
    #X_train = torch.from_numpy(np.array(X_train)).float()
    #y_train = torch.from_numpy(np.array(y_train)).float()
    
    
    inp_size = train_exs[0].x_temporal.shape[1]
    hid_size = args.hidden_size
    out_size = inp_size
    geo_size = train_exs[0].x_geo.shape[0]
    
    n_epochs = args.num_epochs
    batch_size = 10#args.batch_size
    n_batches = int(len(train_exs) / batch_size)
    lr = args.lr
    
    device = args.device
    severity_ind_nn = SeverityIndNN2(inp_size, hid_size, out_size, geo_size)
    severity_ind_nn.to(device)
    
    optimizer = optim.Adam(severity_ind_nn.parameters(), lr=lr)
    loss_func = nn.MSELoss()
    
    for epoch in range(0, n_epochs):
        ex_indices = [i for i in range(0, len(train_exs))]
        random.shuffle(ex_indices)
        total_loss = 0.0
        
        for batch_idx in range(n_batches):
            # print("batch_idx", batch_idx)
            batch_ex_indices = ex_indices[batch_idx * batch_size: (batch_idx + 1) * batch_size]
            batch_loss = 0.0
            
            for idx in batch_ex_indices:
        # for idx in ex_indices:
                X = torch.from_numpy(train_exs[idx].x_temporal).float().to(device)
                geo = torch.from_numpy(train_exs[idx].x_geo).float().to(device)
                y = torch.from_numpy(train_exs[idx].y).float().to(device)
            
                severity_ind_nn.zero_grad()
                prediction = severity_ind_nn.forward(X, geo)
                loss = loss_func(prediction, y)
            
                batch_loss += loss
                total_loss += loss
            
        # batch_loss /= batch_size
            batch_loss.backward() # compute gradients
            #batch_loss.backward()
            optimizer.step() # apply gradients
        print("Total loss on epoch %i: %f" % (epoch, total_loss))
        
    return severity_ind_nn