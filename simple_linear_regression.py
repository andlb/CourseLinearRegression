# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression

class linearRegressionClass(object):    
         
    def readdataset(self,dataset):
        self.dataset = pd.read_csv(dataset)
    
    def setVariableYandX(self): #set variable from the dataset       
        self.x = self.dataset.iloc[:, :-1].values
        self.y = self.dataset.iloc[:, 1].values
    
    def setTrainingSetandTestSet(self):
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.x, self.y, test_size = 1/3, random_state = 0)        

    def setFitLinearRegressor(self):
        self.regressor = LinearRegression()
        self.regressor.fit(self.x_train, self.y_train)
    
    def predict(self):
        self.y_pred = self.regressor.predict(self.x_test)
    
    def showTrainigResult(self):
        plt.scatter(self.x_train, self.y_train,color='red')
        plt.plot(self.x_train, self.regressor.predict(self.x_train), color='blue')
        plt.title('Salary Vs Experience (training set)')
        plt.xlabel('Years of experience')
        plt.ylabel('Salary')
        plt.show()
        
    def showTestResult(self):
        plt.scatter(self.x_test, self.y_test,color='red')
        plt.plot(self.x_train, self.regressor.predict(self.x_train), color='blue')
        plt.title('Salary Vs Experience (test set)')
        plt.xlabel('Years of experience')
        plt.ylabel('Salary')
        plt.show()
    
obj = linearRegressionClass()
obj.readdataset("Salary_Data.csv")
obj.setVariableYandX()
obj.setTrainingSetandTestSet()
obj.setFitLinearRegressor()
obj.predict()
obj.showTrainigResult()
obj.showTestResult()



        
# Importing the dataset
#dataset = pd.read_csv('Salary_Data.csv')
#X = dataset.iloc[:, :-1].values
#y = dataset.iloc[:, 1].values

# Splitting the dataset into the Training set and Test set



# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""




#Visualizando o training set result



#Visualizando o test set result



