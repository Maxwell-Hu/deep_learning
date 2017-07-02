import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


dummy_field = ['Pclass','Sex','Embarked']
file_path = './data/train.csv'


def load_data(filename):
    fp = pd.read_csv(filename)
    for each in dummy_field:
        dummies = pd.get_dummies(fp[each],prefix=each)
        fp = pd.concat([fp,dummies],axis=1)
    fields_to_drop = dummy_field + ['Ticket','Cabin','Name','PassengerId','Survived']
    datas = fp.drop(fields_to_drop,axis=1)
    labels = fp.iloc[:,1]
    train_data,valid_data = datas[:int(len(datas)*train_precent)],datas[int(len(datas)*train_precent):]
    train_labels,valid_labels = labels[:int(len(labels)*train_precent)],labels[int(len(labels)*train_precent):]
    return train_data,train_labels,valid_data,valid_labels


train_precent = 0.8
train_features,train_labels,valid_features,valid_labels = load_data(file_path)
assert len(train_features) == len(train_labels)
assert len(valid_features) == len(valid_labels)


class NeuralNetwork(object):
    def __init__(self,input_nodes,hidden_nodes,output_nodes,learning_rate):
        # Set the number of nodes in input, hidden, output layers.
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        # Initialize weights
        self.weights_input_to_hidden = np.random.normal(0.0, self.input_nodes**-0.5, (self.input_nodes,self.hidden_nodes))
        self.weights_hidden_to_output = np.random.normal(0.0, self.hidden_nodes**-0.5, (self.hidden_nodes, self.output_nodes))

        self.lr = learning_rate

        self.activation = lambda x: 1/1+np.exp(-x)


    def train(self, features, targets):
        '''Train the network on batch of features and targets.

            Arguments
            ---------

            features: 2D array, each row is one data record, each coloum is a feature
            targets : 1D array of target values.

        '''
        n_records = features.shape[0]
        delta_weights_i_h = np.zeros(self.weights_input_to_hidden.shape)
        delta_weights_h_o = np.zeros(self.weights_hidden_to_output.shape)
        for x,y in zip(features,targets):
            hidden_inputs  = np.sum(np.dot(x,self.weights_input_to_hidden),axis=0)
            hidden_outputs = self.activation(hidden_inputs)

            final_inputs = np.dot(hidden_outputs,self.weights_hidden_to_output)
            final_outputs = final_inputs

            error = y - final_outputs

            # Calculate the hidden layer's attribute to the error
            hidden_error = error * self.weights_hidden_to_output

            # Backpropagation
            output_error_term = error
            hidden_error_term = hidden_error

            delta_weights_h_o += error * hidden_outputs
            delta_weights_i_h += hidden_error_term * hidden_ouputs * (1 - hidden_outputs)

            self.weights_hidden_to_output += delta_weights_h_o
            self.weights_input_to_hidden += delta_weights_i_h

    def run(self, features):
         '''Run a forward pass through the network with input features

            Arguments:
            ----------
            features: 1D array of feature value
        '''

        # forward pass
        hidden_inputs = np.dot(features , self.weights_input_to_hidden)
        hidden_outputs = self.activation(hidden_inputs)

        final_inputs = np.dot(hidden_outputs , self.weights_hidden_to_output)
        final_outputs = final_inputs

        return final_outputs

def MSE(y,Y):
    ''' Calculate Mean Squared Error

        Argument:
        --------
        y: prediction values
        Y: target values
    '''
    return np.mean((y-Y)**2)


''' Hyperparamters
'''
epochs = 100
learning_rate = 0.01
hidden_nodes = 20
output_nodes = 1


'''Train the network
'''
n_features = train_features.shape[1]
network = NeuralNetwork(n_features, hidden_nodes, output_nodes, learning_rate)

loss = {'train':[], 'validation':[]}
for i in range(epochs):
    batch = np.random.choice(train_features.index,size=128)
    for record, target in zip(train_features.ix[batch].values,train_labels.ix[batch].values):
        network.train(record,target)

    tain_loss  = MSE(network.run(train_features),train_labels)
    val_loss   = MSE(network.run(valid_features),valid_labels)
    print('Epochs : {}\n Validation loss: {}'.format(i,val_loss))

    loss['train'].append(train_loss)
    loss['validation'].append(val_loss)

''' Plot the changes of loss
'''
plt.plot(loss['train'],label='Training Loss')
plt.plot(loss['validation'],label='Validation Loss')
plt.legeng()
plt.ylim(ymax=1)

