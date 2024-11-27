#%%
import pickle, gzip
import os
import torch
from torch import nn
import torch.functional as func
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt
os.getcwd()
#%%

#%%
torch.random.manual_seed(1234)
with open("C:\pix\ml\hw\cnn\mnist.pkl","rb") as f:
    train_set, valid_set, test_set = pickle.load(f,encoding="latin1")

#%%
seq = nn.Sequential(
            nn.Conv2d(in_channels=1,out_channels=10,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=10,out_channels=10,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2))
out = seq(torch.as_tensor(np.random.random((1,28,28)).astype(np.float32)))
out.shape
# %%
class MnistClassifier(nn.Module):
    def __init__(self, hidden_layer_size = 512):
        super().__init__()
        self.all_layers = nn.Sequential(
            nn.Conv2d(in_channels=1,out_channels=10,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=10,out_channels=10,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten(),
            nn.Linear(490,hidden_layer_size),
            nn.ReLU(),
            nn.Linear(hidden_layer_size,hidden_layer_size),
            nn.ReLU(),
            nn.Linear(hidden_layer_size,10),
            nn.Softmax(dim=1)
        )

    def forward(self,x):
         
        return self.all_layers(x)
    
def train(model, train_X, train_y, valid_X, valid_y,loss_fn, optimizer, epochs = 50, learning_rate=1e-02, batch_size = 500, min_epochs_before_stopping = 30, min_valid_loss_improvement = 0.0025):
    
    train_ds = torch.utils.data.TensorDataset(train_X,train_y)

    train_dl = torch.utils.data.DataLoader(train_ds,batch_size=batch_size,shuffle=True)
    train_loss = []
    train_acc = []
    valid_loss = []
    valid_acc = []
    for epoch in range(epochs):
        model.train()
        epoch_loss = []
        for idx, (X,y) in enumerate(train_dl):
            
            preds = model(X)
            loss = loss_fn(preds,y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            epoch_loss.append(loss.item())
        train_loss.append(np.mean(epoch_loss))
        preds = model(X)
        train_acc.append(accuracy_score(np.argmax(model(X).detach().numpy(),axis=1),y.numpy()))
        with torch.no_grad():
            
            pred_valid = model(valid_X)
            valid_loss.append(loss_fn(pred_valid,valid_y).item())
            valid_acc.append(accuracy_score(np.argmax(pred_valid.detach().numpy(),axis=1),valid_y.numpy()))

            if epoch > min_epochs_before_stopping:
                valid_loss_delta = np.mean(valid_loss[-10:]) - np.mean(valid_loss[-20:-10])
                if valid_loss_delta > 0 or abs(valid_loss_delta) < min_valid_loss_improvement:
                    print(f"Stopping early at epoch {epoch} - validation loss not improving significantly...")
                    break
        print(f"epoch : {epoch}, train loss/acc: {train_loss[-1]:0.4f}/{train_acc[-1] * 100:0.2f}, validation loss/acc: {valid_loss[-1]:0.4f}/{valid_acc[-1] * 100:0.2f}")
    return train_loss,train_acc,valid_loss,valid_acc,epoch

def test(model, test_X, test_y, loss_fn):
    model.eval()
    with torch.no_grad():
        preds = model(test_X)
        test_loss = loss_fn(preds,test_y)
        test_accuracy = accuracy_score(np.argmax(preds.detach().numpy(),axis=1),test_y.numpy())

        print(f"test loss/acc: {test_loss.item():0.4f}/{test_accuracy * 100:0.2f}")
    return test_loss,test_accuracy

# %%

learning_rate = 1e-02 
model = MnistClassifier()
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adagrad(model.parameters(),lr=learning_rate)
train_loss, train_acc,valid_loss,valid_acc,epochs = train(model,
                                                torch.as_tensor(train_set[0].reshape(-1,1,28,28)),
                                                torch.as_tensor(train_set[1]),
                                                torch.as_tensor(valid_set[0].reshape(-1,1,28,28)),
                                                torch.as_tensor(valid_set[1]),
                                                loss_fn,
                                                optimizer)
#%%



#%%
fig,axes = plt.subplots(1,2)
axes[0].plot(range(len(valid_loss)),train_loss,valid_loss)
axes[1].plot(range(len(valid_acc)),train_loss,valid_loss)
plt.show()

#%%
test_loss,test_accuracy = test(model,
                               torch.as_tensor(test_set[0].reshape(-1,1,28,28)),
                               torch.as_tensor(test_set[1]),
                               loss_fn)


# %%
