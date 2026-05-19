# DL- Developing a Recurrent Neural Network Model for Stock Prediction

## AIM
To develop a Recurrent Neural Network (RNN) model for predicting stock prices using historical closing price data.

## Problem Statement and Dataset
Sequential data such as stock prices, weather records, and sensor readings contain important temporal patterns that traditional machine learning models often fail to capture effectively. Accurate prediction of future values from such time-series data is challenging because it requires understanding dependencies between past and present observations.

The problem addressed in this project is to design and implement a Recurrent Neural Network (RNN) model capable of learning these sequential dependencies and performing reliable time-series prediction. The objective is to preprocess the dataset, train an RNN-based deep learning model, and evaluate its performance in forecasting future values with improved accuracy.

This project aims to demonstrate how Deep Learning, specifically RNN architecture, can be used to model sequential patterns, reduce prediction errors, and provide an effective solution for real-world forecasting problems such as stock market analysis, demand prediction, and trend estimation.

### Train Dataset:
![alt text](Output-img/train.png)

### Test Dataset:
![alt text](Output-img/test.png)

## DESIGN STEPS
### STEP 1: 

Load and Preprocess Data

### STEP 2: 

Define RNN Model

### STEP 3: 

Train the Model

### STEP 4: 

Make Predictions on Test Set

### STEP 5: 

Display the prediction.



## PROGRAM

### Name: Krishna Prasad S

### Register Number: 212223230108

```python
# Define RNN Model
class RNNModel(nn.Module):
  def __init__(self, input_size=1,hidden_size=64,num_layers=2,output_size=1):
    super(RNNModel, self).__init__()
    self.rnn = nn.RNN(input_size, hidden_size, num_layers,batch_first=True)
    self.fc = nn.Linear(hidden_size, output_size)
  def forward(self,x):
    out, _= self.rnn(x)
    out=self.fc(out[:,-1,:])
    return out

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Train the Model
def train_model(model, train_loader, criterion, optimizer, epochs=20):
  train_losses = []
  model.train()
  for epoch in range(epochs):
    total_loss = 0
    for x_batch, y_batch in train_loader:
      x_batch, y_batch=x_batch.to(device), y_batch.to(device)
      optimizer.zero_grad() # Clear previous gradients
      outputs = model(x_batch) # Forward pass
      loss = criterion(outputs, y_batch)
      loss.backward() # Backpropagation
      optimizer.step() # Update weights
      total_loss += loss.item()
    train_losses.append(total_loss / len(train_loader))
    print(f'Epoch [{epoch+1}/{epochs}], Loss: {total_loss / len(train_loader) :.4f}')
  # Plot training loss
  print('Name: Krishna Prasad S')
  print('Register Number: 212223230108')
  plt.plot(train_losses, label='Training Loss')
  plt.xlabel('Epoch')
  plt.ylabel('MSE Loss')
  plt.title('Training Loss Over Epochs')
  plt.legend()
  plt.show()


```

### OUTPUT

## Training Loss Over Epochs Plot
![alt text](Output-img/loss.png)

## True Stock Price, Predicted Stock Price vs time
![alt text](Output-img/output.png)

### Predictions
![alt text](Output-img/output-2.png)

## RESULT
Thus, a Recurrent Neural Network (RNN) model for predicting stock prices using historical closing price data has been developed successfully.
