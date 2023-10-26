import deep_convolutional_representation as dcr 

import numpy as np

import statsmodels.api as sm
import statsmodels.formula.api as smf

from sklearn.metrics import mean_absolute_error, r2_score, max_error
from sklearn.metrics import explained_variance_score, mean_squared_error

import matplotlib.pyplot as plt
import matplotlib.pyplot as plt


class load_model():

    def __init__(self, x_train, y_train, x_test, y_test, checkpoint_path: str, path_to_save: str, deeper_model: bool, *args, **kwargs):

        self.x_train = x_train
        self.y_train = y_train

        self.x_test = x_test
        self.y_test = y_test

        self.checkpoint_path = checkpoint_path
        self.path_to_save = path_to_save

        self.deeper_model = deeper_model

        self.init_kernel_size = kwargs.get('init_kernel_size') 


    def plot(self, X, Y, max_value = 1000):

        # Stats
        results = sm.OLS(Y,sm.add_constant(X)).fit()
        
        print(results.summary())


        # Figure
        plt.figure(figsize=(6, 6)) 

        fontsize = 18


        # line of best fit
        linear_fit = np.linspace(0, max_value, max_value)
        
        plt.plot(linear_fit, linear_fit*results.params[1] + results.params[0], '-.', color='tab:blue') #, alpha=0.7)


        # Ideal y=x 
        y = x = np.linspace(0, max_value, max_value)
        
        plt.plot(x, y, '--', color='red', alpha=0.8) 


        # Predicted vs Actual
        plt.plot(X, Y, 'o', markersize=5, color='black', alpha=0.15)

        ticks = np.linspace(0, max_value, 5)
        
        plt.xticks(ticks, fontsize=fontsize)
        plt.yticks(ticks, fontsize=fontsize)
        plt.xlabel('True $\lambda_{max}$ (nm)', fontsize=fontsize)
        plt.ylabel('Predicted $\lambda_{max}$ (nm)', fontsize=fontsize)
        plt.tick_params(axis='both', which='both', labelsize=fontsize, direction="in")
        plt.rcdefaults()


        #Text in figure
        font1 = {'family': 'DejaVu Sans',
                'color':  'red',
                'weight': 'normal',
                'size': fontsize,
                }

        font2 = {'family': 'DejaVu Sans',
                'color':  'tab:blue',
                'weight': 'normal',
                'size': fontsize,
                }

        font3 = {'family': 'DejaVu Sans', 
                'color':  'black',
                'weight': 'normal',
                'size': fontsize,
                }

        plt.show()
        

        print('m = ', results.params[1])
        print('c = ', results.params[0], '\n')

        print('MAE: ', mean_absolute_error(X, Y))
        print('MSE: ', mean_squared_error(X, Y))
        print('RMSE: ', mean_squared_error(X, Y, squared=False))
        print('R-squared: ', r2_score(X, Y))
        print('Max error: ', max_error(X, Y))
        print('Explained_variance_score: ', explained_variance_score(X, Y, multioutput='variance_weighted'))


    
    def evaluate(self):

        CNN = dcr.prepare(self.path_to_save)

        if self.deeper_model == True:
            
            model = CNN.create_model(self.x_train, self.y_train, self.init_kernel_size)
            
        else:
            
            model = CNN.create_simpler_model(self.x_train, self.y_train, self.init_kernel_size)

    
        model.load_weights(self.checkpoint_path)


        # Re-evaluate the model
        loss_train = model.evaluate(self.x_train, self.y_train, verbose=1)

        print("Loss of training data: {:5.2f} nm".format(np.sqrt(loss_train[0])))
        print("RMSE of training data: {:5.2f} nm".format(np.sqrt(loss_train[1])))
        print("MAE of training data: {:5.2f} nm".format(loss_train[2]))

        loss = model.evaluate(self.x_test, self.y_test, verbose=1)

        print("Loss of validation data: {:5.2f} nm".format(np.sqrt(loss[0])))
        print("RMSE of validation data: {:5.2f} nm".format(np.sqrt(loss[1])))
        print("MAE of validation data: {:5.2f} nm".format(loss[2]))


        y_pred = model.predict(self.x_test)

        print()
        print('Plot using validation/test set:')
        self.plot(self.y_test, y_pred)
        
        
        return model, self.y_test, y_pred
