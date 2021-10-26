# -*- coding: utf-8 -*-
"""Tutorial2.ipynb
**Introduction to PyTorch: From Linear Regression to Two-Layer Neural Network**
"""

import numpy as np
import torch
import sklearn.datasets 
import matplotlib.pyplot as plt
import torch.nn as nn
from tqdm import tqdm, trange

"""## Download data
Description of the dataset https://scikit-learn.org/stable/datasets/toy_dataset.html
"""

boston = sklearn.datasets.load_boston()

x1 = boston.data[:,-1]
y1 = boston.target

x = torch.tensor(x1, dtype=torch.float)
y = torch.tensor(y1, dtype=torch.float)
x = x/x.max()

plt.scatter(x, y);
plt.show()

class MyDataset(torch.utils.data.Dataset):
    """
    Our dataset
    """
    def __init__(self, x, y):
        self.x = x/np.max(x)
        self.y = y
        
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        return torch.tensor(self.x[idx], dtype=torch.float).unsqueeze(dim=-1), torch.tensor(self.y[idx], dtype=torch.float).unsqueeze(dim=-1)

dataset = MyDataset(x1, y1)




class MyNN(nn.Module):
  def __init__(self, input_size, hidden_size, output_size):
    super(MyNN, self).__init__()
    self.hidden_size = hidden_size
    self.fc1 = nn.Linear(input_size,hidden_size)
    self.a1 = nn.Sigmoid()
    self.fc = nn.Linear(hidden_size,output_size)

  def forward(self, x):
      out = self.fc1(x)
      out = self.a1(out)
      out = self.fc(out)
      return out

model = MyNN(1,50,1)

#torch.manual_seed(0) # if you wish tofix random numbers

learning_rate = 0.1
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

criterion = nn.MSELoss()

num_epochs = 50

dataloader = torch.utils.data.DataLoader(dataset, batch_size=4)


track_loss = []
for epoch in range(num_epochs):
  iterator = iter(dataloader)
  lossMovingAveraged = 0
  with trange(len(dataloader)) as t:
    for idx in t:
        outputs = model(x_train)                        
        loss = criterion(outputs, y_train)
        loss.backward()                                
        optimizer.step()                               
        optimizer.zero_grad()  

        batch_loss = loss.cpu().detach().item()
        lossMovingAveraged += (batch_loss - lossMovingAveraged) / (idx + 1)

        t.set_description(f"epoch : {epoch}, loss {round(lossMovingAveraged, 3)}")
  track_loss.append(lossMovingAveraged)

plt.plot(track_loss)
plt.show()