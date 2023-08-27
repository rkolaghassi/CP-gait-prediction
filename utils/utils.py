import pandas as pd
import os
from matplotlib import pyplot as plt
from matplotlib.pyplot import figure
from matplotlib import rcParams
from scipy import stats
import scipy
import numpy as np
from tqdm.notebook import tqdm
from torch import nn, optim, strided
import time  
import torch 
import random
import seaborn as sns
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from datetime import date
import pickle
import sys



def set_settings():

    features = ['Hips Flexion-Extension Left',
        'Knees Flexion-Extension Left',
        'Ankles Dorsiflexion-Plantarflexion Left',
        'Hips Flexion-Extension Right',
        'Knees Flexion-Extension Right',
        'Ankles Dorsiflexion-Plantarflexion Right',
        ]

    input_window = 100
    output_window = 1
    stride = 1
    return features, input_window, output_window, stride



def setDevice():
  DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
  return DEVICE



def create_dataframe(train_files, all_features):
    all_data = pd.DataFrame()

    for f in train_files:
        if os.path.exists(f):
            print(f'Extracting data from: {f}')
        file_data = pd.read_csv(f)
        columns = [] #all columne names

        # finds the columns names of interest 
        for idx, _ in enumerate(all_features):
            col_name = list(filter(lambda x: x.startswith(all_features[idx]), list(file_data.columns)))
            columns.extend(col_name)

        # only keep columns of interest
        fltrd_data = file_data[columns]

        #Add patient ID to dataframe
        fltrd_data.insert(0, 'Patient ID', value=[f[:-4]]*fltrd_data.shape[0])

        #Remove data at the beggining and in between trials (correspond to Trial value of 0)
        data = fltrd_data.drop(fltrd_data[fltrd_data.Trial == 0].index)

        #Concatenate trials 
        all_data = pd.concat([all_data, data], axis=0)

    return all_data




def window_generator(sequence, input_window, output_window, stride, features, labels):
    """
    Trims the input sequence from leading and trailing zeros, then generates an array with input windows and another array for the corresponding output windows
    Args:
        sequence: (np.array, float32) columns are features while rows are time points
        features: (list, string) column names
        input_window: (int)
        stride (int): the value the input window shifts along the sequence 
    Returns:

    """
    # shortest_seqLen = float('inf')

    # f_zeros = [] #array that stores the number of leading zeros for each feature
    b_zeros = [] #array that stores the number of trailing zeros for each feacture 

    for f in features:
        # trim the leading and training zeros
        # f_zeros.append(sequence[:,labels[f]].shape[0] - np.trim_zeros(sequence[:,labels[f]], 'f').shape[0]) #forward zeros
        b_zeros.append(sequence[:,labels[f]].shape[0] - np.trim_zeros(sequence[:,labels[f]], 'b').shape[0]) #backward zeros

    # max_f_zeros = max(f_zeros) #find the maximum number of leading zeros
    max_b_zeros = max(b_zeros) #find the maximum number of trailing/backward zeros 

    #total sequence length minus max leading and trailing zeros 
    trimmed_seqLen = sequence[:,0].shape[0] - (max_b_zeros)    
    print(f'trimmed_seqLen: {trimmed_seqLen}')

    # Slides are the number of times the input window can scan the sequence 
    # Using the equation that calculates the number of outputs as in convolution  (W – F + 2P) / S + 1, W=input image width, F=filter width, P=padding, S=stride
    # The width of the image is taken as the number of time steps in the sequence, corresponding to the length of any TRIMMED column in the data 
    slides = ((trimmed_seqLen - (input_window+output_window)) // stride) + 1
    print(f"number of slides is: {slides}")

    # Calculating the first index of each of the output sequences (first index always f_zeros as its always shifted to start with the first non-zero element)
    seq_indicies = (np.arange(slides) * stride)

    if slides <= 0:
        raise ValueError("input window and output window length are greater than sequence length, check their values")

    # Creates an zero numpy array to store the samples in 
    X_values = np.zeros((len(seq_indicies) , input_window, len(features)))
    Y_values = np.zeros((len(seq_indicies), output_window, len(features)))

    # Loop through the features, then loop through the list of sequence indicies needed for input and output windows 
    for j, feature in enumerate(features):

        for i, idx in enumerate(seq_indicies):
            X_values[i, :, j] = sequence[idx:idx+input_window, labels[feature]]
            Y_values[i, :, j] = sequence[idx+input_window:idx+input_window + output_window, labels[feature]]

    return X_values, Y_values 






def normalise_fit(data):
    '''
    Normalises X_train and output scales to use for normalising testing data and de-normalising output
    Args:
        prenormalised_data should be a 3-dimensional np.array 
    
    Output:
        normalised_data: np.array with the same shape as input data, but normalised
        scaling_factors: np.array with 2 rows index 0 for min_val, index 1 for max_value and nfeatures columns 

    '''
    
    if data.ndim != 3:
        raise ValueError("this function can only normalise 3-dimensional inputs")

    normalised_data = np.zeros(data.shape, dtype=np.float32) 

    features = data.shape[-1]
    scaling_factors = np.zeros((2, features))

    for feature in range(features):
        scaling_factors[0, feature], scaling_factors[1, feature] =  data[:,:,feature].min(), data[:,:,feature].max() #index 0 is for min, index 1 for max (applies to all features)
        normalised_data[:,:,feature] = (data[:,:,feature] - data[:,:,feature].min())/ (data[:,:,feature].max() - data[:,:,feature].min())

        # scaling_factors[0, feature], scaling_factors[1, feature] =  -90, 90 #index 0 is for min, index 1 for max (applies to all features)
        # normalised_data[:,:,feature] = (data[:,:,feature] - (-90))/ ((90) - (-90))

    return normalised_data, scaling_factors





def normalise_transform(data, scaling_factors):
    '''
    Normalise data based on the scaling factors used to normalised the training data 
    '''

    transformed_data = np.zeros(data.shape, dtype=np.float32)
    if data.shape[-1] > 1:
        for feature in range(data.shape[-1]): #loop over the number of features
            min_val = scaling_factors[0, feature] #get minimum value
            max_val = scaling_factors[1, feature] # get maximum value 

            transformed_data[:,:,feature] = (data[:,:,feature] - min_val) / (max_val - min_val)
    
    # in case there is one feature only where scaling_factor shape (2,1)
    else:
        min_val = scaling_factors[0]
        max_val = scaling_factors[1]
        transformed_data[:,:,0] = (data[:,:,0] - min_val) / (max_val - min_val)
    
    return transformed_data




def denormalise(data, scaling_factors):
    '''
    De-normalise data of the model
    '''
    denormalised_data = np.zeros(data.shape,  dtype=np.float32)

    for feature in range(data.shape[-1]):
        min_val = scaling_factors[0, feature]
        max_val = scaling_factors[1, feature]

        denormalised_data[:,:,feature] = ((max_val - min_val) * data[:,:,feature]) + min_val
    
    return denormalised_data





def create_dataframe(train_files, all_features):

    all_data = pd.DataFrame()

    # try:
    for f in train_files:

        if os.path.exists(f):
            print(f'Extracting data from: {f}')

        file_data = pd.read_csv(f)
        columns = [] #all columne names

        # finds the columns names of interest 
        for idx, _ in enumerate(all_features):
            col_name = list(filter(lambda x: x.startswith(all_features[idx]), list(file_data.columns)))
            columns.extend(col_name)

        # only keep columns of interest
        fltrd_data = file_data[columns]

        #Add patient ID to dataframe
        fltrd_data.insert(0, 'Patient ID', value=[f[:-4]]*fltrd_data.shape[0])

        #Remove data at the beggining and in between trials (correspond to Trial value of 0)
        data = fltrd_data.drop(fltrd_data[fltrd_data.Trial == 0].index)

        #Concatenate trials into a big array 
        all_data = pd.concat([all_data, data], axis=0)
    
    # except Exception:
    #     print(f'Error with file: {f}')

    return all_data



def count_nsamples(data):
    n_samples=0
    for p in data['Patient ID'].unique(): #loop over patients
        for i in data['Trial'].unique(): #loop over trial
            d = data[(data['Patient ID'] == p) & (data['Trial'] == i)]
            if d.empty:
                #print('DataFrame is empty!')
                #print(f'Trail {i} does not exist in {p}')
                continue
            else:
                print(f'For patient: {p}, trial: {i}, there are: {len(d)} time-points')
                n_samples += 1

    print(f'\nThere are {n_samples} samples')
    return(n_samples)





# Creates dataset object that gets individual samples for training/testing so that the Dataloader can generate batches
class gaitDataset(Dataset):
    def __init__(self, x, y):
        self.x = x 
        self.y = y 

    def __len__(self):
        return self.x.shape[0]
    
    def __getitem__(self, index):
        X_sample = self.x[index, :, :]
        Y_sample = self.y[index, :, :]
        return X_sample, Y_sample


    

 
# Training the LSTM model using a loss function and a optimiser
def train_LSTM(model, train_dataloader, val_dataloader, num_epochs, learning_rate, device):
    loss_function = nn.MSELoss(reduction='mean')
    optimiser = torch.optim.Adam(model.parameters(), lr = learning_rate)

    train_loss = np.zeros(num_epochs)
    val_loss = np.zeros(num_epochs)

    for epoch in tqdm(range(num_epochs)):
        # Loop over batch values 
        runningLoss_train = 0. 

        for idx, (batch_inputs, batch_targets) in enumerate(train_dataloader):
            
            # Save batch on GPU
            batch_inputs, batch_targets = batch_inputs.to(device), batch_targets.to(device)

            model.train() 

            #set gradients to zero
            optimiser.zero_grad()
            preds = model(batch_inputs)
        
            loss = loss_function(preds, batch_targets)
            

            loss.backward()
            optimiser.step()
            runningLoss_train += loss.item()

        train_loss[epoch] = runningLoss_train / len(train_dataloader)

        print(f"Epoch: [{epoch + 1}/{num_epochs}]", f"Training loss: {runningLoss_train/len(train_dataloader)}")

        # Evaluate on validation set

        model.eval() #evaluating the model, stops process such as dropout etc. 
        runningLoss_val = 0.

        with torch.no_grad(): # makes sure gradient is not stored 
            for idx, (batch_inputs, batch_targets) in enumerate(val_dataloader):
                
                batch_inputs, batch_targets = batch_inputs.to(device), batch_targets.to(device)
                optimiser.zero_grad() 
                preds = model(batch_inputs)

                loss = loss_function(preds, batch_targets)
                runningLoss_val += loss.item()

        val_loss[epoch] = runningLoss_val/len(val_dataloader)

        print(f"Epoch: [{epoch + 1}/{num_epochs}]", f"Validation loss: {runningLoss_val/len(val_dataloader)}")

    return train_loss, val_loss




# Testing function 
def test_LSTM(model, dataloader, device):
    loss_function = nn.MSELoss(reduction='mean')
    model.eval()
    actual_output, pred_output = [], []
    running_loss = 0. 
    
    with torch.no_grad():
        for idx, (batch_inputs, batch_targets) in tqdm(enumerate(dataloader)):

            batch_inputs, batch_targets = batch_inputs.to(device), batch_targets.to(device)

        

            batch_preds = model(batch_inputs)
            loss = loss_function(batch_preds, batch_targets)
            running_loss += loss.item()
            actual_output.append(batch_targets)
            pred_output.append(batch_preds)

        total_loss = running_loss / len(dataloader)

        actual_output_tensor = torch.vstack(actual_output)
        pred_output_tensor = torch.vstack(pred_output)
    
    return pred_output_tensor, actual_output_tensor, total_loss



def mse_loss(preds, targets, reduction = 'mean', format='torch'):

    
    if format == 'torch': #default option
        loss = 1/(preds.shape[0]*preds.shape[1]*preds.shape[2]) * torch.sum((targets - preds) ** 2)
        std = torch.std()
        
        if reduction == 'sum':
            loss = torch.sum((targets - preds) ** 2)
    
    if format == 'np':
        loss = 1/(preds.shape[0]*preds.shape[1]*preds.shape[2]) * np.sum((targets - preds) ** 2)
        std = np.std(((targets - preds) ** 2).reshape(-1,1).squeeze())
        
        if reduction == 'sum':
            loss = np.sum((targets - preds) ** 2)

    return loss, std



#mae 
def mae_loss(preds, targets, reduction = 'mean', format='torch'):

    if format == 'torch': #default option
        loss = 1/(preds.shape[0]*preds.shape[1]*preds.shape[2]) * torch.sum(torch.abs(targets - preds))

        
        if reduction == 'sum':
            loss = torch.sum(torch.abs(targets - preds))
    
    if format == 'np':
        loss = 1/(preds.shape[0]*preds.shape[1]*preds.shape[2]) * np.sum(np.absolute(targets - preds))  
        
        std = np.std(np.abs(targets - preds).reshape(-1,1).squeeze())

        if reduction == 'sum':
            loss = np.sum(np.absolute(targets - preds))

    return loss, std


def pd_to_np_converter(data, n_samples, features):
    #create a numpy array that stores the data for export
    sample_ID = []
    # patients = 2
    # n_trials = 10
    # # samples = patients * n_trials
    data_store = np.zeros((n_samples, 2000, len(features)), dtype=np.float32)
    i = 0

    for p in data['Patient ID'].unique(): #loop over patients 
        for t in data['Trial'].unique(): #loop over trials starting with trials 1 to trial 9 (inclusive)
            pd_array = data[(data['Patient ID'] == p) & (data['Trial'] == t)]
            if pd_array.empty:
                continue
                # print('DataFrame is empty!')
                # print(f'Trail {t} does not exist in {p}')
            else:
                np_array = pd_array.to_numpy()
                data_store[i, :np_array.shape[0], :] = np_array[:,3:] 
                sample_ID.append(p+ ' Ts'+str(t)) 
                i +=1

    return pd_array.columns, data_store

def window_generator_fltrd(sequence, input_window, output_window, stride, features, labels):
    """
    Trims the input sequence from leading and trailing zeros, then generates an array with input windows and another array for the corresponding output windows
    Args:
        sequence: (np.array, float32) columns are features while rows are time points
        features: (list, strin~g) column names
        input_window: (int)
        stride (int): the value the input window shifts along the sequence 
    Returns:

    """
    # shortest_seqLen = float('inf')

    # f_zeros = [] #array that stores the number of leading zeros for each feature
    b_zeros = [] #array that stores the number of trailing zeros for each feacture 

    for f in features:
        # trim the leading and training zeros
        # f_zeros.append(sequence[:,labels[f]].shape[0] - np.trim_zeros(sequence[:,labels[f]], 'f').shape[0]) #forward zeros
        b_zeros.append(sequence[:,labels[f]].shape[0] - np.trim_zeros(sequence[:,labels[f]], 'b').shape[0]) #backward zeros

    # max_f_zeros = max(f_zeros) #find the maximum number of leading zeros
    max_b_zeros = max(b_zeros) #find the maximum number of trailing/backward zeros 

    #total sequence length minus max leading and trailing zeros 
    trimmed_seqLen = sequence[:,0].shape[0] - (max_b_zeros)
    trimmed_seqLen_reduced = trimmed_seqLen - 300 #reducing sequence size to remove the first and last 200 timesteps which may contain errors   
    print(f'trimmed_seqLen: {trimmed_seqLen}')
    print(f'trimmed_seqLen_reduced: {trimmed_seqLen_reduced}')


    # Slides are the number of times the input window can scan the sequence 
    # Using the equation that calculates the number of outputs as in convolution  (W – F + 2P) / S + 1, W=input image width, F=filter width, P=padding, S=stride
    # The width of the image is taken as the number of time steps in the sequence, corresponding to the length of any TRIMMED column in the data 
    slides = ((trimmed_seqLen_reduced - (input_window+output_window)) // stride) + 1
    print(f"number of slides is: {slides}")

    # Calculating the first index of each of the output sequences (first index always f_zeros as its always shifted to start with the first non-zero element)
    seq_indicies = (np.arange(slides) * stride) + 150

    if slides <= 0:
        raise ValueError("input window and output window length are greater than sequence length, check their values")

    # Creates an zero numpy array to store the samples in 
    X_values = np.zeros((len(seq_indicies) , input_window, len(features)))
    Y_values = np.zeros((len(seq_indicies), output_window, len(features)))

    # Loop through the features, then loop through the list of sequence indicies needed for input and output windows 
    for j, feature in enumerate(features):
        for i, idx in enumerate(seq_indicies):
            X_values[i, :, j] = sequence[idx:idx+input_window, labels[feature]]
            Y_values[i, :, j] = sequence[idx+input_window:idx+input_window + output_window, labels[feature]]

    return X_values, Y_values 

def window_generator_lt_fltrd(sequence, input_window, future_window, stride, features, labels): #window gernerator long term fltrd (creats a validation window up to 200 timesteps in advance to measure error on long term future predictions)
    """
    Trims the input sequence from leading and trailing zeros, then generates an array with input windows and another array for the corresponding output windows
    Args:
        sequence: (np.array, float32) columns are features while rows are time points
        features: (list, string) column names
        input_window: (int)
        stride (int): the value the input window shifts along the sequence 
    Returns:

    """
    b_zeros = [] #array that stores the number of trailing zeros for each feacture 

    for f in features:
        # trim the leading and training zeros
        b_zeros.append(sequence[:,labels[f]].shape[0] - np.trim_zeros(sequence[:,labels[f]], 'b').shape[0]) #backward zeros

    max_b_zeros = max(b_zeros) #find the maximum number of trailing/backward zeros 

    fltrd_samples = 2 * 150 #remove 100 timesteps from the beggining and ending of the entire sequence
    # lt_len = 200 # number of timesteps to predict in the future based on a single input window (to be used in measuring errors based on prediction input)
    
    #total sequence length minus max leading and trailing zeros 
    trimmed_seqLen = sequence[:,0].shape[0] - (max_b_zeros)
    trimmed_seqLen_reduced = trimmed_seqLen - (fltrd_samples) # (- fltrd_samples is done to reduce sequence size to remove the first and last 150 timesteps which may contain errors since they corresponding to beggining and ending of the trials 
    print(f'trimmed_seqLen: {trimmed_seqLen}')
    print(f'trimmed_seqLen_reduced: {trimmed_seqLen_reduced}')


    # Slides are the number of times the input window can scan the sequence 
    # Using the equation that calculates the number of outputs as in convolution  (W – F + 2P) / S + 1, W=input image width, F=filter width, P=padding, S=stride
    # The width of the image is taken as the number of time steps in the sequence, corresponding to the length of any TRIMMED column in the data 
    slides = ((trimmed_seqLen_reduced - (input_window+future_window)) // stride) + 1
    print(f"number of slides is: {slides}")

    # Calculating the first index of each of the output sequences (first index always f_zeros as its always shifted to start with the first non-zero element)
    seq_indicies = (np.arange(slides) * stride) + 150

    if slides <= 0:
        raise ValueError("input window and output window length are greater than sequence length, check their values")
        # return None 

    # Creates an zero numpy array to store the samples in 
    X_values = np.zeros((len(seq_indicies) , input_window, len(features)))
    Y_values = np.zeros((len(seq_indicies), future_window, len(features)))

    # Loop through the features, then loop through the list of sequence indicies needed for input and output windows 
    for j, feature in enumerate(features):

        for i, idx in enumerate(seq_indicies):
            X_values[i, :, j] = sequence[idx:idx+input_window, labels[feature]]
            Y_values[i, :, j] = sequence[idx+input_window:idx+input_window + future_window, labels[feature]]

    return X_values, Y_values 