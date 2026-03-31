import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as datetime
from collections import Counter
import time
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import shap

db = pd.read_csv('modes.csv')


# %% Training and test data split: Splitting the dataset into training, validation, and test data set using 60:20:20 split for train: validation: test.

from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import classification_report

db.head()

#X = db.iloc[:,4:] 
X = db[['xmin', 'ymin', 'zmin', 'xmean', 'ymean', 'zmean', 'xstd', 'ystd', 'zstd']].values
#you can play with using less features to see the impact on the accuracy 
#X = data[['xmean', 'ymean', 'zmean', 'xstd', 'ystd', 'zstd']].values
#X = data[['xstd', 'zstd']].values
y = db['Class']
#label encoding is done as model accepts only numeric values
# so strings need to be converted into labels
LE = preprocessing.LabelEncoder()
LE.fit(y)
y = LE.transform(y)

#splitting dataset into train, validation and test data
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state = 0)
X_train,X_val,y_train,y_val = train_test_split(X_train,y_train,test_size=0.25,random_state = 0)

# Scale data
scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)


#Output the number of data points in training, validation, and test dataset.
print("Datapoints in Training set:",len(X_train))
print("Datapoints in validation set:",len(X_val))
print("Datapoints in Test set:",len(X_test))

# Convert the y variable into one-hot encoding - basically the true label will be 1 and all others will be assigned to 0
def one_hot(y, num_classes):
    return np.eye(num_classes)[y]

y_train_oh = one_hot(y_train, len(np.unique(y)))
y_val_oh = one_hot(y_val, len(np.unique(y)))
y_test_oh = one_hot(y_test, len(np.unique(y)))


# %% -------------  PART A -------------------
#Train NN models to obtain the accuracy on the test data using your training and validation data.
#Simple Neural Network from Scratch with One Hidden Layer 
#Forward and backward propogation are to be implemented in detail 
#Missing parts that you need to implement to answer the question are indicated below (with "Implement!")


#activation functions 
def tanh(x):
    return np.tanh(x)

def relu(x):
    return np.maximum(x, 0)

def softmax(x):
    x = x - np.max(x, axis=0, keepdims=True)
    expX = np.exp(x)
    return expX / np.sum(expX, axis=0, keepdims=True)

#derivatives - activation functions - to be used in backward progation of errors
def derivative_tanh(x):
    return (1 - np.power(np.tanh(x), 2))

def derivative_relu(x):
    return np.array(x > 0, dtype = np.float32)

#n_x number of features aka input variables
#n_h number of neurons in the hidden layer 
#n_y number of classes 


#definition of parameters (theta) between each input in the input layer, each neuron in the hidden layer, and output in the output layer
def initialize_parameters(n_x, n_h, n_y):
    theta_1 = np.random.randn(n_h, n_x)#*0.1  scaling used for the case of RELU 
    theta0_1 = np.zeros((n_h, 1))
    
    theta_2 = np.random.randn(n_y, n_h)#*0.1  scaling used for the case of RELU 
    theta0_2 = np.zeros((n_y, 1))
    
    parameters = {
        "theta_1" : theta_1,  
        "theta0_1" : theta0_1, #bias / intercept from the input layer
        "theta_2" : theta_2,
        "theta0_2" : theta0_2 #bias / intercept from the hidden layer
    }
    
    return parameters

def forward_propagation(x, parameters):
    
    theta_1 = parameters['theta_1']
    theta0_1 = parameters['theta0_1']
    theta_2 = parameters['theta_2']
    theta0_2 = parameters['theta0_2']
    
    # !!!!!!! IMPLEMENT !!!!!!!!!!!!!!!!
    #linear combination of first set of parameters and the inputs 
    #x1 = .........
    x1= np.dot(theta_1, x) + theta0_1 #solution
    xh1 = relu(x1) #activation function
    #linear combination of the second set of parameteres and the output of the hidden layer
    #x2 = ........
    x2=  np.dot(theta_2, xh1) + theta0_2 #solution
    xh2 = softmax(x2) #softmax function at the output layer for the classification task
    #xh2 is prediction
    
    forward_cache = {
        "x1" : x1,
        "xh1" : xh1,
        "x2" : x2,
        "xh2" : xh2
    }
    
    return forward_cache
#%%
def cost_function(xh2, y):
    m = y.shape[1]
    #cross-entropy loss (also in your slides - for multiclass classification with softmax)
    cost = -(1/m)*np.sum(y*np.log(xh2))
    return cost

def backward_prop(x, y, parameters, forward_cache):
    
    theta_1 = parameters['theta_1']
    theta0_1 = parameters['theta0_1']
    theta_2 = parameters['theta_2']
    theta0_2 = parameters['theta0_2']
    
    xh1 = forward_cache['xh1']
    xh2 = forward_cache['xh2']
    
    m = x.shape[1]
    
    dx2 = (xh2 - y)  #output layer with the softmax - partial derivative with respect to x2, this is given to you and have quite some derivations behind 
    
    #partial derivative with respect to the second set of parameters based on the error above
    dtheta_2 = (1/m)*np.dot(dx2, xh1.T)
    dtheta0_2 = (1/m)*np.sum(dx2, axis = 1, keepdims = True)
    
    #error propagated to the hidden layer 
    dx1 = np.dot(theta_2.T, dx2)*derivative_relu(xh1)  #needs to be tailored to the chosen activation function at the hidden layer
    
    # !!!!!!! IMPLEMENT !!!!!!!!!!!!!!!!
    #partial derivative with respect to the first set of parameters based on the error above
    dtheta_1 = (1/m)*np.dot(dx1, x.T) #solution use x.T instead of xh1.T
    dtheta0_1 = (1/m)*np.sum(dx1, axis = 1, keepdims = True)
    
    gradients = {
        "dtheta_1" : dtheta_1,
        "dtheta0_1" : dtheta0_1,
        "dtheta_2" : dtheta_2,
        "dtheta0_2" : dtheta0_2
    }
    
    return gradients

def update_parameters(parameters, gradients, learning_rate):
    
    theta_1 = parameters['theta_1']
    theta0_1 = parameters['theta0_1']
    theta_2 = parameters['theta_2']
    theta0_2 = parameters['theta0_2']
    
    dtheta_1 = gradients['dtheta_1']
    dtheta0_1 = gradients['dtheta0_1']
    dtheta_2 = gradients['dtheta_2']
    dtheta0_2 = gradients['dtheta0_2']
    
    #update of the first and second set of parameters based on the partial derivatives (gradients) and the learning rate
    theta_1 = theta_1 - learning_rate*dtheta_1
    theta0_1 = theta0_1 - learning_rate*dtheta0_1
    theta_2 = theta_2 - learning_rate*dtheta_2
    theta0_2 = theta0_2 - learning_rate*dtheta0_2
    
    parameters = {
        "theta_1" : theta_1,
        "theta0_1" : theta0_1,
        "theta_2" : theta_2,
        "theta0_2" : theta0_2
    }
    
    return parameters

def NN_singleHiddenLayer(x, y, n_h, learning_rate, iterations):
    
    n_x = x.shape[0]
    n_y = y.shape[0]
    
    cost_list = []
    
    parameters = initialize_parameters(n_x, n_h, n_y)
    
    for i in range(iterations):
        
        forward_cache = forward_propagation(x, parameters)
        
        cost = cost_function(forward_cache['xh2'], y)
        
        gradients = backward_prop(x, y, parameters, forward_cache)
        
        parameters = update_parameters(parameters, gradients, learning_rate)
        
        cost_list.append(cost)
        
        if(i%(iterations/10) == 0):
            print("Cost after", i, "iterations is :", cost)
        
    return parameters, cost_list

iterations = 1000
n_h = 10
learning_rate = 0.05
Parameters, Cost_list = NN_singleHiddenLayer(X_train.T, y_train_oh.T, n_h = n_h, learning_rate = learning_rate, iterations = iterations)

t = np.arange(0, iterations)
plt.plot(t, Cost_list)
plt.show()

#accuracy function based on a test/validation dataset 
def accuracy(inp, labels, parameters):
    forward_cache = forward_propagation(inp, parameters)
    y_out = forward_cache['xh2']   # containes propabilities with shape(6, 1)
    y_out = np.argmax(y_out, 0)  # 0 represents row wise 
    labels = np.argmax(labels, 0)
    acc = np.mean(y_out == labels)*100
    
    return acc

print("Accuracy on the Train Dataset", round(accuracy(X_train.T, y_train_oh.T, Parameters),2), "%")
print("Accuracy on the Validation Dataset", round(accuracy(X_val.T, y_val_oh.T, Parameters), 2), "%")



# %% -------------  PART B -------------------
#Now Train NN models using SKLEARN 
#Logistic regression (that can handle multi class classification) is provided for you. 

#Missing parts that you need to implement to answer the question are indicated below (with "Implement!")


from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

train_logreg = LogisticRegression(random_state=0,max_iter = 200).fit(X_train,y_train)

# !!!!!!! IMPLEMENT !!!!!!!!!!!!!!!!
# Train a Neural Network
# own solution train_nn = MLPClassifier(random_state=1, max_iter=300)
train_nn = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(30,30),random_state=1,max_iter=500)


train_nn.fit(X_train,y_train)

pred_logreg = train_logreg.predict(X_val)
print("For Logistic Regression: ")
print(classification_report(y_val, pred_logreg))
print ("Accuracy of logistic regression on the initial data is: ",accuracy_score(pred_logreg,y_val))

# !!!!!!! IMPLEMENT !!!!!!!!!!!!!!!!
#Predict based on the trained Neural Network using the validation data
#NOTE: You should reach a NN which has a better accuracy than the logistic regression, if  not revisit the specification of your NN

pred_nn = train_nn.predict(X_val)
print("For Neural Network: ")
print(classification_report(y_val, pred_nn))
print ("Accuracy of NN on the initial data is: ",accuracy_score(pred_nn,y_val))


# %% -------------  PART C -------------------
#We can also extract features such as minute, hour and day from timestamp column as it was not used till now and try to improve the above accuracies

#Feature Extraction from Timestamp column
db.head()
db['Start Time'] = db['Start Time'].astype('datetime64[ns]')
db['hour'] = db['Start Time'].dt.hour
db['minute'] = db['Start Time'].dt.minute
db['day'] = db['Start Time'].dt.day
db.head()

#Again training the classifiers
X = db.iloc[:,4:]
y = db['Class']
#label encoding is done as model accepts only numeric values
# so strings need to be converted into labels
LE = preprocessing.LabelEncoder()
LE.fit(y)
y = LE.transform(y)

#splitting dataset into train, validation and test data
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state = 0)
X_train,X_val,y_train,y_val = train_test_split(X_train,y_train,test_size=0.25,random_state = 0)

# Scale data
scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# Convert the y variable into one-hot encoding - basically the true label will be 1 and all others will be assigned to 0
def one_hot(y, num_classes):
    return np.eye(num_classes)[y]

y_train_oh = one_hot(y_train, len(np.unique(y)))
y_val_oh = one_hot(y_val, len(np.unique(y)))
y_test_oh = one_hot(y_test, len(np.unique(y)))

train_logreg2 = LogisticRegression(random_state=0,max_iter = 200).fit(X_train,y_train)

# !!!!!!! IMPLEMENT !!!!!!!!!!!!!!!!
#Train a NN with more features as explained above 
# own train_nn2 = MLPClassifier(random_state=0, max_iter=200)
train_nn2 = MLPClassifier(solver='adam', activation='relu', alpha=1e-5, hidden_layer_sizes=(128,64),random_state=1,max_iter=1500, learning_rate_init=0.001)
train_nn2.fit(X_train,y_train)

pred_logreg2 = train_logreg2.predict(X_val)
print("For Logistic Regression: ")
print(classification_report(y_val, pred_logreg2))
print ("Accuracy of logistic regression on the extended data is: ",accuracy_score(pred_logreg2,y_val))

# !!!!!!! IMPLEMENT !!!!!!!!!!!!!!!!
# Predict based on the trained NN using the validation data
#NOTE: Again you should reach a NN which has a better accuracy than the logistic regression, if  not revisit the specification of your NN


pred_nn2 = train_nn2.predict(X_val)
print("For Neural Network: ")
print(classification_report(y_val, pred_nn2))
print ("Accuracy of NN on the extended data is: ",accuracy_score(pred_nn2,y_val))

# %% -------------  PART D -------------------
#Accuracy of the models should increase by using additional features
#Pick the one with the highest accuracy and apply it to the test data. 

# !!!!!!! IMPLEMENT !!!!!!!!!!!!!!!!
#NOTE you should reach an accuracy of at least 80% so revisit your models if you cannot reach that. 

final_res = train_nn2.predict(X_test) #solution
print ("Accuracy of the chosen Classifier is: ",accuracy_score(final_res,y_test))
# %%
