import torch
from torch import nn
import time
import copy
import torch.utils.data as Data
import numpy as np
import matplotlib.pyplot as plt


# data
data = np.loadtxt('../data/iris.data')

def load_data(data,k_train=0.9,BATCH_SIZE_TRA=1,BATCH_SIZE_VAL=1,SHUFFLE_BOOL_TRA=False,
              SHUFFLE_BOOL_VAL=False,NUM_WORKERS_TRA=0,NUM_WORKERS_VAL=0):
    data_length = len(data)

    x_train = torch.FloatTensor(np.array(data[:k_train * data_length,:4]))
    y_train = torch.LongTensor(np.array(data[:k_train * data_length,4]))

    x_val = torch.FloatTensor(np.array(data[k_train * data_length:,:4]))
    y_val = torch.LongTensor(np.array(data[k_train * data_length:,4]))

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
        return out, h_state


def construct_model_opt(INPUT_SIZE,HIDDEN_SIZE,OUTPUT_SIZE,LR=1e-3,OPT = 'Adam',WEIGHT_DECAY=0,
                        LOSS_NAME = 'crossentropy',MODEL = 'RNN'):

    if MODEL == 'RNN':
        model = RNN(INPUT_SIZE,HIDDEN_SIZE,OUTPUT_SIZE,NUM_LAYERS=1,NONLINEARITY='tanh',
                 BIAS_RNN_BOOL=True,BATCH_FIRST=True,DROPOUT_PRO=0,BIDIRECTIONAL_BOOL=False)
    # else:
    #     model = LSTM(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE, NUM_LAYERS=1, NONLINEARITY='tanh',
    #                 BIAS_RNN_BOOL=True, BATCH_FIRST=True, DROPOUT_PRO=0, BIDIRECTIONAL_BOOL=False)

    if OPT == 'Adagrad':
        optimizer = torch.optim.Adagrad(model.parameters(),lr = LR, weight_decay=WEIGHT_DECAY)
    elif OPT == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    if LOSS_NAME == 'MSELoss':
        criterion = nn.MSELoss()
    else:
        criterion = nn.CrossEntropyLoss()

    return model, optimizer, criterion


# train
def train_model(model,train_loader,val_loader,criterion,optimizer,
                PATH,num_epochs=25,CUDA_ID="0",isClassfier=True):
    if torch.cuda.is_available():
        device = torch.device("cuda:"+CUDA_ID)

    else:
        device = torch.device("cpu")
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    since = time.time()
    val_acc_history = []
    best_model_wts = copy.deepcopy(model.state_dict)  # if model is so complex, and metric is not acc,
                                                      # not recommend this, may occupy much memory.
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        model.train()
        running_loss = 0.0
        running_corrects = 0
        h_state = 0
        for step_0, (train_x, train_y) in enumerate(train_loader):
            output_tra, h_state= model(train_x,h_state) #  output
            h_state = h_state.data
            loss_tra = criterion(output_tra, train_y)  # cross entropy loss
            optimizer.zero_grad()  # clear gradients for this training step
            loss_tra.backward()  # backpropagation, compute gradients
            optimizer.step()  # apply gradients

            _, tra_preds = torch.max(output_tra, 1)
            running_loss += loss_tra.item() * train_x.size(0)
            if isClassfier:
                running_corrects += torch.sum(tra_preds == train_y.data)

        epoch_tra_loss = running_loss / len(train_loader.dataset)
        epoch_tra_acc = running_corrects.double() / len(train_loader.dataset)
        print('Train Loss: {:.4f} Acc: {:.4f}'.format(epoch_tra_loss, epoch_tra_acc))

        running_loss = 0.0
        running_corrects = 0
        h_state = 0
        model.eval()
        for step_1, (val_x, val_y) in enumerate(val_loader):
            output_val, _ = model(val_x,h_state) #  output
            loss = criterion(output_val, val_y)  # cross entropy loss

            _, val_preds = torch.max(output_val, 1)
            running_loss += loss.item() * val_x.size(0)
            if isClassfier:
                running_corrects += torch.sum(val_preds == val_y.data)

        epoch_val_loss = running_loss / len(val_loader.dataset)
        if isClassfier:
            epoch_val_acc = running_corrects.double() / len(val_loader.dataset)
            print('Val Loss: {:.4f} Acc: {:.4f}'.format(epoch_val_loss, epoch_val_acc))
            # deep copy the model
            if  epoch_val_acc > best_acc:
                best_acc = epoch_val_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            val_acc_history.append(epoch_val_acc)


    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    # torch.save(the_model, PATH)
    torch.save(model, PATH)
    # return model, val_acc_history


# test


# save whole model
def load_model_test(PATH,data):
    # Model class must be defined somewhere
    model = torch.load(PATH)
    model.eval()

    # 1. simple, not many samples





# save parameters
def load_param_test(model,TheModelClass,PATH):


    ###
    model = TheModelClass(*args, **kwargs)
    model.load_state_dict(torch.load(PATH))
    model.eval()


if __name__ =='__main__':
    torch.save(model.state_dict(), '/model_save/model_params.pkl')


