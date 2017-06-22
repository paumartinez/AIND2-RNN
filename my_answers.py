import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import keras


# TODO: fill out the function below that transforms the input series 
# and window-size into a set of input/output pairs for use with our RNN model
def window_transform_series(series,window_size):
    # containers for input/output pairs
    X = []
    y = []

    for i in range(0,len(series)-window_size-1):
        y.append(series[i+window_size]) # output saved
        aux = [] # auxiliar variable to save the input
        
        for j in range(i,window_size+i):
            aux.append(series[j])   
        
        X.append(aux) # input saved

    # reshape each 
    X = np.asarray(X)
    X.shape = (np.shape(X)[0:2])
    y = np.asarray(y)
    y.shape = (len(y),1)
    
    return X,y

# TODO: build an RNN to perform regression on our time series input/output data
def build_part1_RNN(step_size, window_size):
    
    # given - fix random seed - so we can all reproduce the same results on our default time series
    np.random.seed(0)

    # TODO: build an RNN to perform regression on our time series input/output data
    model = Sequential()  # secuential model
    model.add(LSTM(5, input_shape=(window_size, 1))) #adding a LSTM layer with 5 hidden units
    model.add(Dense(1)) # adding a fully connected layer with 1 unit

    # build model using keras documentation recommended optimizer initialization
    optimizer = keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)

    # compile the model
    model.compile(loss='mean_squared_error', optimizer=optimizer)


### TODO: list all unique characters in the text and remove any non-english ones
def clean_text(text):
    # find all unique characters in the text
    all_characters = ''.join(set(text))  # string with all the characters present on text

    # remove as many non-english characters and character sequences as you can 

    # string with the allowed characters
    persist_characters = ' qwertyuiopasdfghjklzxcvbnmQWERTYUIOPASDFGHJKLZXCVBNM.,:;"()!?'
    for i in all_characters:
        if i not in persist_characters:
            # remove from text all characters that are not in persist_characters
            text = text.replace(i,' ')  
            
    # shorten any extra dead space created above
    text = text.replace('  ',' ')


### TODO: fill out the function below that transforms the input text and window-size into a set of input/output pairs for use with our RNN model
def window_transform_text(text,window_size,step_size):
    # containers for input/output pairs
    inputs = []
    outputs = []
    
    for i in np.arange(0,len(text)-window_size-1,step_size):
        # The FOR sentence takes in account the step_size

        outputs.append(text[i+window_size]) # output saved
        aux = '' # auxiliar variable to save the input (string)
        for j in range(i,window_size+i):
            aux = aux + text[j]
        inputs.append(aux) # input saved
    
    return inputs,outputs
