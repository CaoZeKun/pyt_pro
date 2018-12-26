import torch
from torch import nn
import time
import copy
import torch.utils.data as Data
import numpy as np
import matplotlib.pyplot as plt


# data
data = np.loadtxt('../data/iris.data')

def load_data(data,BATCH_SIZE_TRA=1,BATCH_SIZE_VAL=1,SHUFFLE_BOOL_TRA=False,
              SHUFFLE_BOOL_VAL=False,NUM_WORKERS_TRA=0,NUM_WORKERS_VAL=0):
    data_length = len(data)

    x_train = torch.FloatTensor(np.array(data[:0.9 * data_length,:4]))
    y_train = torch.LongTensor(np.array(data[:0.9 * data_length,4]))

    x_val = torch.FloatTensor(np.array(data[0.9 * data_length:,:4]))
    y_val = torch.LongTensor(np.array(data[0.9 * data_length:,4]))

    train_data = Data.TensorDataset(x_train, y_train)
    train_loader = torch.utils.data.DataLoader(dataset = train_data,
                                               batch_size = BATCH_SIZE_TRA,
                                               shuffle = SHUFFLE_BOOL_TRA,
                                               num_workers = NUM_WORKERS_TRA,)
    val_data = Data.TensorDataset(x_val, y_val)
    val_loader = torch.utils.data.DataLoader(dataset = val_data,
                                             batch_size = BATCH_SIZE_VAL,
                                             shuffle = SHUFFLE_BOOL_VAL,
                                             num_workers = NUM_WORKERS_VAL,)

    return train_loader, val_loader


# model
class RNN(nn.Module):
    def __init__(self,INPUT_SIZE,HIDDEN_SIZE,OUTPUT_SIZE,NUM_LAYERS=1,NONLINEARITY='tanh',
                 BIAS_RNN_BOOL=True,BATCH_FIRST=True,DROPOUT_PRO=0,BIDIRECTIONAL_BOOL=False):
        super(RNN,self).__init__()
        self.rnn = nn.RNN(
            input_size = INPUT_SIZE,
            hidden_size = HIDDEN_SIZE,
            num_layers = NUM_LAYERS,
            nonlinearity = NONLINEARITY,  #  'tanh' | 'relu
            bias = BIAS_RNN_BOOL,
            batch_first = BATCH_FIRST,  # data_format (batch, seq, feature)
            dropout = DROPOUT_PRO,
            bidirectional = BIDIRECTIONAL_BOOL,
        )
        self.out = nn.Linear(HIDDEN_SIZE,OUTPUT_SIZE,bias=BIAS_RNN_BOOL)

    def forward(self,x,h_state):
        # x (batch, seq , feature)
        # h_state (num_layers * num_directions, batch, hidden_size)
        # r_out (batch, seq , num_directions * hidden_size)
        r_out, h_state = self.rnn(x,h_state)
        # choose r_out at the last time step
        out = self.out(r_out[:,-1,:])
        return out


def construct_model_opt(INPUT_SIZE,HIDDEN_SIZE,OUTPUT_SIZE,LR=1e-3,OPT = 'Adam',WEIGHT_DECAY=0,
                        LOSS_NAME = 'crossentropy',MODEL = 'RNN'):

    if MODEL == 'RNN':
        rnn = RNN(INPUT_SIZE,HIDDEN_SIZE,OUTPUT_SIZE,NUM_LAYERS=1,NONLINEARITY='tanh',
                 BIAS_RNN_BOOL=True,BATCH_FIRST=True,DROPOUT_PRO=0,BIDIRECTIONAL_BOOL=False)

    if OPT == 'Adagrad':
        optimizer = torch.optim.Adagrad(rnn.parameters(),lr = LR, weight_decay=WEIGHT_DECAY)
    elif OPT == 'SGD':
        optimizer = torch.optim.SGD(rnn.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    else:
        optimizer = torch.optim.Adam(rnn.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    if LOSS_NAME == 'MSELoss':
        loss_fun = nn.MSELoss()
    else:
        loss_fun = nn.CrossEntropyLoss()

    return rnn, optimizer, loss_fun




# train
def train_model(model,dataloaders,criterion,optimizer,num_epochs=25,is_inception=False):
    since = time.time()
    val_acc_history = []
    best_model_wts = copy.deepcopy(model.state_dict)
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    if is_inception and phase == 'train':
                        outputs, aux_outputs = model(inputs)
                        loss1 = criterion(outputs,labels)
                        loss2 = criterion(aux_outputs,labels)
                        loss = loss1 + 0.4 * loss2
                    else:
                        outputs = model(inputs)
                        loss = criterion(outputs,labels)

                    _, preds = torch.max(outputs,1)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val'
                val_acc_history.append(epoch_acc)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m{:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    model.load_state_dict(best_model_wts)
    return model, val_acc_history


# test



if __name__ =='__main__':

