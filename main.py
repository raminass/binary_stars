import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import timeit
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, utils
from torch.utils.data import random_split
from torch import Tensor

from models import *
from mlxtend.preprocessing import standardize


##### Set device to GPU or CPU #####
def setdevice():
    if args.device == "gpu" and torch.cuda.is_available():
        print("Model will be training on the GPU.\n")
        args.device = torch.device('cuda')
    elif args.device == "gpu":
        print("No GPU detected. Falling back to CPU.\n")
        args.device = torch.device('cpu')
    else:
        print("Model will be training on the CPU.\n")
        args.device = torch.device('cpu')

##### List of parameters in data #####
'''physical_params = ['fi',
                    'list_vsini_1',
                    'list_vsini_2',
                    'list_m_1',
                    'list_m_2',
                    'list_a_1',
                    'list_a_2',
                    'list_t_1',
                    'list_t_2',
                    'list_log_g_1',
                    'list_log_g_2',
                    'list_l_1',
                    'list_l_2']'''

################################################################################
##################### Torch Datasets and Helper Funcs ##########################
################################################################################

class StarDataset(Dataset):
    # load the dataset
    def __init__(self, paths, labels, normalize=True):
        
        #Empty files for creating x, y lists
        list_total_X=[]
        list_total_y=[]

        # Iterate through dataset files
        for num in paths:
          # Read file
          path="./data/{}.npz".format(num)
          data = np.load(path)

          # store the inputs and outputs
          self.scale_param=1

          # Save X, append to list
          file_X = data['fi']
          list_total_X.append(file_X)

          # Read all relevant y, append to list 
          list_y = []
          for label in labels:
            list_y.append(data[label])
          file_y = np.column_stack(list_y)
          list_total_y.append(file_y)

        # Turn lists into np
        total_X = np.row_stack(list_total_X)
        total_y = np.row_stack(list_total_y)

        # ensure input data is floats
        self.X = total_X.astype('float32')
        # label encode target and ensure the values are floats
        self.y = total_y.astype('float32')
        self.y = self.y.reshape((-1, len(labels)))

        # Create new dimension, with opposite order of stars
        new_order = []
        for i in range(len(labels)):
          if i%2==0:
            new_order.append(i+1)
          else:
            new_order.append(i-1)
        y_alt = self.y[:,new_order]

        # New y will be stacked values of first option, second option
        self.y = np.stack((self.y, y_alt), axis=-1)
        if normalize:
          self.y[:,:,0], self.scale_param = standardize(self.y[:,:,0],return_params=True)
          self.y[:,:,1] = standardize(self.y[:,:,1])

    def get_scale_params(self):
        return self.scale_param 

    # number of rows in the dataset
    def __len__(self):
        return len(self.X)

    # get a row at an index
    def __getitem__(self, idx):
        return [self.X[idx], self.y[idx]]

    # get indexes for train and test rows
    def get_splits(self, n_test=0.2, seed_value=42):
        # determine sizes
        test_size = round(n_test * len(self.X))
        valid_size = round(n_test * len(self.X))
        train_size = len(self.X) - test_size - valid_size
        # calculate the split
        return random_split(self, [train_size, valid_size, test_size], generator=torch.Generator().manual_seed(seed_value))

######################### Prepare the Torch dataset ############################
def prepare_data(paths, labels, normalize=True):
    # load the dataset
    dataset = StarDataset(paths, labels, normalize)
    # calculate split
    train, valid, test = dataset.get_splits()
    print(f"Creating dataset from {len(dataset)} cases")
    print(f"Split into {len(train)} train, {len(valid)} test, and {len(test)} test")
    # prepare data loaders
    train_dl = DataLoader(train, batch_size=32, shuffle=True)
    test_dl = DataLoader(test, batch_size=1024, shuffle=False)
    valid_dl = DataLoader(valid, batch_size=1024, shuffle=False)
    return train_dl, valid_dl, test_dl, dataset.get_scale_params()

################################################################################
##################### Loss function and Model Evaluations ######################
################################################################################

##################### Loss function - bespoke for Start_new dataset ############
def weighted_mse_loss(inputs, target, weights=[1]):

    # Create array of weights (if desired - default is no weight whange)
    weight = np.array(weights)

    # Adjust weight and input dimensions for broadcasting
    weight = torch.tensor(weight).reshape(1, weight.shape[0], 1)
    inputs = inputs.unsqueeze(dim=-1)

    # Calculate the mean for each option (Star A first then Star B, and the reverse)
    options = torch.sum((weight * (inputs - target) ** 2), dim=1)

    # Take the minimum of the options
    minimum, _ = torch.min(options, dim=-1)

    # Return the minimum, averaged
    return torch.mean(minimum)


########################## Evaluate Model ######################################
def evaluate_model(data, model, wordy=False):
    model.eval()
    with torch.no_grad():
        losses = []
        for x, y in data:
            scores = model(x)
            losses.append(weighted_mse_loss(scores, y))
            if wordy:
              param_len = scores.shape[1]
              for i in range(0, param_len, 2):
                print("Predictions: ", scores[0,i:i+2].tolist())
                print("Actual: ", y[0,i:i+2,0].tolist())
    return sum(losses)/len(losses)

def evaluate_model_denormalize_2_from_12(data, model, scale_params, j=0, wordy=False):
    std_dev = np.reshape(scale_params['stds'], (1,-1))
    avg = np.reshape(scale_params['avgs'], (1,-1))
    param_len = std_dev.shape[1]
    model.eval()
    with torch.no_grad():
        losses = []
        for x, y in data:
            scores = model(x)
            scores_unnormalized = scores*std_dev+avg
            scores_unnormalized = scores_unnormalized[:, 2*j:2*j+2]
            if wordy:
              #for i in range(0, param_len, 2):
                print("Predictions: ", scores_unnormalized[0,:].tolist())
                print("Actual: ", y[0,2*j:2*j+2,0].tolist())
            losses.append(weighted_mse_loss(scores_unnormalized, y[:, 2*j:2*j+2]))
    return sum(losses)/len(losses)

def evaluate_model_denormalize(data, model, scale_params, wordy=False):
    std_dev = np.reshape(scale_params['stds'], (1,-1))
    avg = np.reshape(scale_params['avgs'], (1,-1))
    param_len = std_dev.shape[1]
    model.eval()
    params = ['vsini', 'metallicity', 'alpha', 'temperature', 'log_g', 'luminosity']
    with torch.no_grad():
        losses = []
        for x, y in data:
            scores = model(x)
            scores_unnormalized = scores*std_dev+avg
            if wordy:
              j=0
              for i in range(0, param_len, 2):
                print(f"{params[j]} predictions: ", scores_unnormalized[0,i:i+2].tolist())
                print(f"{params[j]} actual: ", y[0,i:i+2,0].tolist())
                j+=1
            losses.append(weighted_mse_loss(scores_unnormalized, y))
            print("")
    return sum(losses)/len(losses)


def eval_comp_contributions(data, model):
    model.eval()
    with torch.no_grad():
        losses = []
        
        for x, y in data:
            scores = model(x)
            for i in range(0, 12, 2):
              losses.append(weighted_mse_loss(scores[i:i+2], y[i:i+2]))
    return sum(losses)/len(losses)

############################ Post_analysis #####################################
def post_analysis(data, model, criterion, cols):
    idx = [physical_params.index(col)-1 for col in cols]
    stacked_inputs = []
    stacked_tagets = []
    losses = {}
    with torch.no_grad():
        for x, y in data:
            stacked_inputs.append(x)
            stacked_tagets.append(y)
        inputs = torch.cat(stacked_inputs)
        targets = torch.cat(stacked_tagets)
        scores = model(inputs)
        # losses = {col:criterion(scores[:, physical_params.index(col)-1], targets[:, physical_params.index(col)-1])  for col in cols}
        losses = {col:weighted_mse_loss(scores[:, physical_params.index(col)-1], targets[:, physical_params.index(col)-1])  for col in cols}

    return losses

################################################################################
##################### Model Training ###########################################
################################################################################

############## Main function to train models ###################################
def train(data, model, epochs, lr, wd=0, name='', schedule=False):
    trn, vld, tst = data
    tic = timeit.default_timer()
    test_loss_best = 10e30
    trn_loss = []
    tst_loss = []
    vld_loss = []
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    if schedule:
      scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1, verbose=True)
    print(f"Starting training for model: {name}.\n")
    for epoch in range(epochs):
        model.train()
        
        for i, (inputs, targets) in enumerate(trn):
            # clear the gradients
            optimizer.zero_grad()
            # compute the model output
            yhat = model(inputs)
            # calculate loss
            #loss = criterion(yhat, targets)
            loss = weighted_mse_loss(yhat, targets)
            # credit assignment
            loss.backward()
            # update model weights
            optimizer.step()

        model.eval()
        tst_loss.append(evaluate_model(tst, model))
        trn_loss.append(evaluate_model(trn, model))
        vld_loss.append(evaluate_model(vld, model))

        if schedule:
          #print("Learning rate: {}".format(lr))
          scheduler.step()

        print("Epoch : {:d} || Train set loss : {:.3f}".format(
            epoch+1, trn_loss[-1]))
        print("Epoch : {:d} || Validation set loss : {:.3f}".format(
            epoch+1, vld_loss[-1]))
        print("Epoch : {:d} || Test set loss : {:.3f}".format(
            epoch+1, tst_loss[-1]))
        
        test_loss = tst_loss[-1]
        if test_loss < test_loss_best:
          test_loss_best = test_loss
          print("Saving model")
          torch.save(model.state_dict(), f"./state_dicts/{name}_model.pt")
        print("*************************************************\n")
    print("Training is over.")
    return trn_loss, tst_loss, vld_loss


def plot_convergence(trn_loss, vld_loss, tst_loss, title=''):
    fig = plt.figure(figsize=(16, 6))
    plt.plot(np.arange(1, len(trn_loss)+1),
             trn_loss, label='Train Loss')
    plt.plot(np.arange(1, len(vld_loss)+1),
             vld_loss, label='Validation Loss')
    plt.plot(np.arange(1, len(tst_loss)+1),
             tst_loss, label='Test Loss')
    plt.xlabel("EPOCHS")
    plt.ylabel('Loss')
    plt.title(title + ' Convergence Graph')
    plt.legend()
    plt.grid()
    sns.despine(left=True)
    fig.show()
    return fig

################################################################################
##################### Other Helper Functions ###################################
################################################################################

def model_comparison(params, path, learning_rate_a=4e-5, learning_rate_b=4e-5,epochs=20):
  # LOAD AND RUN NORMALIZED MODEL
  # Select parameters for 2-factor model
  name_model = '{}_and_{}_normalized'.format(params[0], params[1])
  print("RUNNING MODEL {}".format(name_model))

  # Load data
  trn_norm, vld_norm, tst_norm, scale_params = prepare_data(path, params)

  # Save scale pasams
  np.save(f"./scale_params/{name_model}_scale_params.npy", scale_params)

  # Run model, track loss, save best model and losses
  data = (trn_norm, vld_norm, tst_norm)
  lr=learning_rate_a
  model = MLP_final(10197, len(params))
  trn_loss_norm, tst_loss_norm, vld_loss_norm = train(data, model, epochs, lr, name=name_model)
  np.save(f"./losses/{name_model}_trn_loss.npy", trn_loss_norm)
  np.save(f"./losses/{name_model}_tst_loss.npy", tst_loss_norm)
  np.save(f"./losses/{name_model}_vld_loss.npy", vld_loss_norm)

  # Plot convergence
  plot_convergence(trn_loss_norm, vld_loss_norm, tst_loss_norm, title=name_model)

  # LOAD AND RUN UNNORMALIZED MODEL
  # Load data
  trn, vld, tst, _ = prepare_data(path, params, normalize=False)
  name_model_2 = '{}_and_{}_NOT_normalized'.format(params[0], params[1])
  print("RUNNING MODEL {}".format(name_model_2))

  # Run model, track loss, save model and losses
  data = (trn, vld, tst)
  lr=learning_rate_b
  model = MLP_final(10197, len(params))
  trn_loss, tst_loss, vld_loss = train(data, model, epochs, lr, name=name_model_2)
  np.save(f"./losses/{name_model_2}_trn_loss.npy", trn_loss)
  np.save(f"./losses/{name_model_2}_tst_loss.npy", tst_loss)
  np.save(f"./losses/{name_model_2}_vld_loss.npy", vld_loss)

  # Plot convergence
  plot_convergence(trn_loss, vld_loss, tst_loss, title=name_model_2)