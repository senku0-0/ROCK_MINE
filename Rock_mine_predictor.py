# We are going to use Logistic Regression to train the model
# Because logistic regression works well with binary classification problems
# Binary Classification is a type of classification problem where the output can take only two possible values
# Work flow
# Dataset --> Data Preprocessing --> Train Test Split --> Model Training(Logistic Regression) --> Model Evaluation
import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split # train_test_split is a function that splits the data into training and testing sets
from sklearn.linear_model import LogisticRegression # LogisticRegression is a class that implements logistic regression
from sklearn.metrics import accuracy_score # accuracy_score is a function that calculates the accuracy of the model
# Data set 
a = pd.read_csv('sonar data.csv', header=None) # header none means assigning column header numeric values 
df = pd.DataFrame(a)

# Analysis the data
print(df.head()) # print the first 5 rows of the dataframe
print(df.shape) # print the shape of the dataframe
print(df.isnull().sum()) # print the count of null values in each column
# there are no null values in the dataset
# 207 rows and 61 columns

# finding types of unique values in the tartget column
# R --> Rock(96) and M --> Mine(111)
print(df[60].value_counts()) # print the count of unique values in the R column

print(df.describe()) # describe the statistical data of the dataframe

# Dataset is good to go
# Seprating feature and target
X = df.drop(columns=60,axis=1) # hold features
Y = df[60] # hold target

X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2,stratify=Y, random_state=1) # split the data into training and testing sets
# startify is used to split the data in such a way that the proportion of the target variable is maintained in both the training and testing sets basically divide the data equally in training and testing sets
# random_state is used to set the seed for the random number generator so that the results are reproducible
# test_size is used to define the testing data size which is 20% in above case

# Model Training 
model = LogisticRegression() # create an object of the LogisticRegression class
model.fit(X_train,Y_train) # fit the model on the training data
# Model Evaluation
# accuracy on training data
X_train_prediction = model.predict(X_train) # predict the target variable on the training data
training_data_accuracy = accuracy_score(X_train_prediction,Y_train) # calculate the accuracy of the model on the training data
print('Accuracy on training data:', training_data_accuracy) # print the accuracy of the model on the training data

# accuracy on Testing data
X_test_prediction = model.predict(X_test) # predict the target variable on the training data
testing_data_accuracy = accuracy_score(X_test_prediction,Y_test) # calculate the accuracy of the model on the training data
print('Accuracy on testing data:', testing_data_accuracy) # print the accuracy of the model on the training data

# Now testing the model with a custom input

input_data = np.array([0.0210,0.0121,0.0203,0.1036,0.1675,0.0418,0.0723,0.0828,0.0494,0.0686,0.1125,0.1741,0.2710,0.3087,0.3575,0.4998,
                       0.6011,0.6470,0.8067,0.9008,0.8906,0.9338,1.0000,0.9102,0.8496,0.7867,0.7688,0.7718,0.6268,0.4301,0.2077,0.1198,0.1660,0.2618,
                       0.3862,0.3958,0.3248,0.2302,0.3250,0.4022,0.4344,0.4008,0.3370,0.2518,0.2101,0.1181,0.1150,0.0550,0.0293,0.0183,0.0104,0.0117,0.0101,0.0061,
                       0.0031,0.0099,0.0080,0.0107,0.0161,0.0133])
input_data_reshaped = input_data.reshape(1,-1) # reshape the input data to match the shape of the training data
# -1 means that the number of columns is unknown and 1 is the number of rows 
# if we give the value of 2 and 3 then it will turn the array into  2 rows and 3 columns

prediction = model.predict(input_data_reshaped) # predict the target variable on the input data
print(prediction) # print the prediction