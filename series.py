import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#importando a tabela com o pandas
df = pd.read_csv('data/vazoes_C_60855000.csv', delimiter=';')
#print(df)

colunaMedia = df[['Media']]
#print(colunaMedia)




meses = 60
training = colunaMedia.iloc[-meses:] #as n últimas
with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
    #print(training)
    pass

#plt.plot(training, "r-")
#plt.show()


#Separando as entradas e saidas do treinamento da rede...
inicio = 0
fim = 12
n = meses - fim
training_input1 = np.empty([n, 12])
training_output = np.zeros([n, 1])


for i in range(n): #quantos meses serão pegos
    training_input = np.append(training.iloc[inicio:fim], [])
    #print("\n", training_input) #print to check
    for j in range(12): #tamanho da janela
        training_input1[i,j] = training_input[j]
    #print(training.iloc[i+1].Media)
    training_output[i] = training.iloc[fim].Media
    inicio = inicio + 1
    fim = fim + 1

#print("")
#pd.set_option('display.float_format', lambda x: '%.3f' % x)
#print("\n", training_input1, training_input1.shape)
#print("\n", training_output, training_output.shape)


class Net(torch.nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Net, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.fc1 = torch.nn.Linear(self.input_size, self.hidden_size)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(self.hidden_size, 1)
        self.sigmoid = torch.nn.Sigmoid()
    def forward(self, x):
        hidden = self.fc1(x)
        sig = self.sigmoid(hidden)
        output = self.fc2(sig)
        output = self.sigmoid(output)
        return output

#convert to tensors
#print(type(training_input1))
#print(type(training_output))

training_input = torch.from_numpy(training_input1)
training_output = torch.from_numpy(training_output)

training_input = training_input.float() / 1000
training_output = training_output.squeeze().float() / 1000

#print(type(training_input))
#print(type(training_output))
#print(training_input)
#print(training_output)

#print(torch.max(training_input))
#print(torch.max(training_output))


input_size = training_input.size()[1] # number of features selected
hidden_size = 30 # number of nodes/neurons in the hidden layer
model = Net(input_size, hidden_size) # create the model
criterion = torch.nn.MSELoss() # works for non-binary classification

# without momentum parameter
optimizer = torch.optim.SGD(model.parameters(), lr = 0.5)
#with momentum parameter
#optimizer = torch.optim.SGD(model.parameters(), lr = 0.9, momentum=0.2)

#print(input_size)
#print(hidden_size)
#print(model)

#print(type(training_input))

model.eval()
#model = model.float()

y_pred = model(training_input)
before_train = criterion(y_pred.squeeze(), training_output)
print('Test loss before training' , before_train.item())


model.train()
epochs = 10000
errors = []
for epoch in range(epochs):
    optimizer.zero_grad()
    # Forward pass
    y_pred = model(training_input)
    # Compute Loss
    loss = criterion(y_pred.squeeze(), training_output)
    errors.append(loss.item())
    print('Epoch {}: train loss: {}'.format(epoch, loss.item()))
    # Backward pass
    loss.backward()
    optimizer.step()


print("FIM DE TREINAMENTO")

#RODAR A REDE COM DADOS NUNCA VISTO PELA REDE
#y_pred = model(test_input)
#loss = criterion(y_pred.squeeze(), test_output)
#Plotar no gráfico
#y_pred
#test_output


print(y_pred.T)
print(training_output)
errors = np.array(errors)
plt.figure(figsize=(15, 5))
plt.subplot(1, 2, 1)
plt.plot(errors, 'r-')

plt.subplot(1, 2, 2)
plt.plot(y_pred.detach().numpy(), 'b-')
plt.plot(training_output.numpy(), 'g-')

plt.show()
