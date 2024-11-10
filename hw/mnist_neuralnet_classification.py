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
torch.random.manual_seed(1234)
with open(".\mnist\mnist.pkl","rb") as f:
    train_set, valid_set, test_set = pickle.load(f,encoding="latin1")
# %%
class MnistClassifier(nn.Module):
    def __init__(self, hidden_layer_size = 512):
        super().__init__()
        self.input_layer = nn.Linear(28 * 28,hidden_layer_size)
        self.hidden_layer_1 = nn.Linear(hidden_layer_size,hidden_layer_size)
        self.output_layer = nn.Linear(hidden_layer_size,10)

    def forward(self,x):
        x = self.input_layer(x)
        x = nn.ReLU()(x)
        x = self.hidden_layer_1(x)
        x = nn.ReLU()(x)
        x = self.output_layer(x)
        return nn.Softmax(dim=1)(x)
    
def train(model, train_X, train_y, valid_X, valid_y,loss_fn, optimizer, epochs = 20, learning_rate=1e-03, batch_size = 250):
    
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

        print(f"epoch : {epoch}, train loss/acc: {train_loss[-1]:0.4f}/{train_acc[-1] * 100:0.2f}, validation loss/acc: {valid_loss[-1]:0.4f}/{valid_acc[-1] * 100:0.2f}")
    return train_loss,train_acc,valid_loss,valid_acc

def test(model, test_X, test_y, loss_fn):
    model.eval()
    with torch.no_grad():
        preds = model(test_X)
        test_loss = loss_fn(preds,test_y)
        test_accuracy = accuracy_score(np.argmax(preds.detach().numpy(),axis=1),test_y.numpy())

        print(f"test loss/acc: {test_loss.item():0.4f}/{test_accuracy * 100:0.2f}")
    return test_loss,test_accuracy

# %%
epochs = 50
learning_rate = 1e-03
model = MnistClassifier(64)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adagrad(model.parameters(),lr=learning_rate)

hidden_layer_size_choices = [64,128,256,512,1024]
validation_results = []
for hidden_layer_size in hidden_layer_size_choices:
    print(f"Training/validating model with hidden layer size {hidden_layer_size}...")
    model = MnistClassifier(hidden_layer_size)
    train_loss, train_acc,valid_loss,valid_acc = train(model,
                                                    torch.as_tensor(train_set[0]),
                                                    torch.as_tensor(train_set[1]),
                                                    torch.as_tensor(valid_set[0]),
                                                    torch.as_tensor(valid_set[1]),
                                                    loss_fn,
                                                    optimizer,
                                                    epochs=epochs)
    validation_results.append((valid_acc,model,hidden_layer_size,train_loss, valid_loss, train_acc))

best_model = sorted(validation_results,key=lambda tup: tup[0], reverse=True)[0]
print(f"Best model is with hidden layer size of {best_model[2]} and validation accuracy {best_model[0]*100:0.2f}")


#%%
fig,axes = plt.subplots(1,2)
axes[0].plot(range(epochs),best_model[3],best_model[4])
axes[1].plot(range(epochs),best_model[5],best_model[0])
plt.show()

#%%
test_loss,test_accuracy = test(best_model[1],
                               torch.as_tensor(test_set[0]),
                               torch.as_tensor(test_set[1]),
                               loss_fn)

