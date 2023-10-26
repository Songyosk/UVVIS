"""
Module to train, optimize and evulate ML model 

Author: Son Gyo Jung
Email: sgj13@cam.ac.uk
"""  
import os   
import numpy as np                                                                                                                                                                               
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.preprocessing import label_binarize
from sklearn.model_selection import train_test_split
from sklearn.metrics import multilabel_confusion_matrix, roc_curve, roc_auc_score, max_error, \
                            auc, f1_score, classification_report, recall_score, precision_recall_curve, \
                            balanced_accuracy_score, confusion_matrix, accuracy_score, average_precision_score, \
                            hamming_loss, matthews_corrcoef, mean_squared_error, mean_absolute_error, r2_score, \
                            plot_confusion_matrix, explained_variance_score


from lightgbm.sklearn import LGBMClassifier, LGBMRegressor
from xgboost import XGBClassifier, XGBRegressor

from skopt import forest_minimize, gbrt_minimize, gp_minimize, dummy_minimize
from sklearn.model_selection import cross_val_score
from skopt.utils import use_named_args
from skopt.space import Real, Integer
from skopt import dump, load
from skopt.plots import plot_convergence, plot_objective, plot_evaluations
from skopt import dump, load

import statsmodels.api as sm
import statsmodels.formula.api as smf

from itertools import cycle


class optimization():
    """
    Optimize and evulate ML model 

    args: 
    (a) path_to_train_data (type:str); location of the training data 
    (b) path_to_test_data (type:str); location of the test data 
    (c) path_to_features (type:str); location of the features to use
    (d) path_to_save (type:str); location to save new data files
    (e) problem (type:str); whether it is a classification or regression problem
    
    return: performance evaluation of ML model
    """
    def __init__(self, path_to_train_data, path_to_test_data, path_to_features, path_to_save, problem, *args, **kwargs):
        self.path_to_save = path_to_save
        self.sample_train = joblib.load(path_to_train_data)  
        self.sample_test = joblib.load(path_to_test_data)
        self.RFE_features = joblib.load(path_to_features)

        #self.features = self.sample_train.columns.values[:-1]
        self.target = self.sample_train.columns.values[-1]

        #self.sample_train, self.sample_val = train_test_split(self.sample_train, test_size=0.1, random_state=42)

        print('Name of target column: ', self.target)
        print('No. of exploratory features: ', len(self.RFE_features))

        self.problem = problem
        self.target_classes = kwargs.get('target_classes')
        self.estimator = kwargs.get('estimator')


    def base_model(self, boosting_method):
        """
        Choose baseline model

        args: boosting_method
        return: baseline model
        """
        self.boosting_method = boosting_method

        if self.problem == 'classification':
            if self.boosting_method == 'lightGBM':
                self.estimator = LGBMClassifier(
                                                boosting_type='gbdt',
                                                objective='multiclass',
                                                random_state=42,
                                                importance_type='gain',
                                                max_depth=-1
                                                )


            elif self.boosting_method == 'XGBoost':
                    self.estimator = XGBClassifier(
                                                    objective='multi:softprob',
                                                    booster='gbtree',
                                                    random_state=42,
                                                    importance_type='total_gain'
                                                    )

        elif self.problem == 'regression':
            if self.boosting_method == 'lightGBM':
                self.estimator = LGBMRegressor(
                                                boosting_type ='gbdt',
                                                random_state=42,
                                                importance_type='gain',
                                                max_depth=-1
                                                )


            elif self.boosting_method == 'XGBoost':
                self.estimator = XGBClassifier(
                                                objective='reg:squarederror',
                                                booster='gbtree',
                                                random_state=42,
                                                importance_type='total_gain'
                                                )

        return self.estimator


    def set_hyperparameters(self, *args, **kwargs):
        """
        Define the hyperparameter space where optimization will be conducted

        args: x0 (type: list) - list of initial guess (optional)
        return: hyperparameter space 
        """
        self.x0 = kwargs.get('x0') # initial guess

        self.space = [
                    Real(0.01, 0.3, name='learning_rate', prior='log-uniform'),
                    Integer(100, 1000, name='n_estimators'),
                    Integer(10, 100, name='num_leaves')

                    # Other parameters can be added e.g.
                    # Integer(10, 100, name='max_depth'),
                    # Real(1, 10, name='min_child_weight', prior='uniform'), 
                    ]

        self.hyperparameters = [
                                'learning_rate',
                                'n_estimators',
                                'num_leaves'
                                ]

        return self.hyperparameters, self.space


    def run(self, optimization_method):
        """
        Execute optimization using one of the methods

        args: optimization_method (type:str); choose one of the following :- dummy_minimize, gp_minimize, gbrt_minimize, forest_minimize
        return: value of the hyperparameters 
        """
        @use_named_args(self.space)
        def objective(**params):
            """
            Define the objective function
            """
            # Performance metric to consider
            if self.problem == 'classification':
                scoring = 'f1_weighted'

            elif self.problem == 'regression':
                scoring = 'neg_root_mean_squared_error'

            self.estimator.set_params(**params)
            
            print('\n', params, '\n')
            
            score = -np.mean(cross_val_score(self.estimator, 
                                            #self.sample_val[self.RFE_features], 
                                            #self.sample_val[self.target], 
                                            self.sample_train[self.RFE_features], 
                                            self.sample_train[self.target], 
                                            cv = 2, 
                                            n_jobs = -1, 
                                            scoring = scoring
                                            )
                            )
            
            print('Score: ', score, '\n')
            return score


        self.optimization_method = optimization_method

        if self.optimization_method == 'random_search':
            opt_method = dummy_minimize

        elif self.optimization_method == 'bayesian':
            opt_method = gp_minimize

        elif self.optimization_method == 'gradient_bossted_trees':
            opt_method = gbrt_minimize

        elif self.optimization_method == 'decision_trees':
            opt_method = forest_minimize

        if self.x0 is not None:
            self.opt = opt_method(
                                func = objective, 
                                dimensions = self.space, 
                                n_calls = 100, 
                                #random_state = 42, 
                                verbose = 1,
                                x0 = [self.x0]
                                ) 
        else:
            self.opt = opt_method(
                    func = objective, 
                    dimensions = self.space, 
                    n_calls = 100, 
                    #random_state = 42, 
                    verbose = 1
                    ) 

        self.values = list()

        print('\n', '*** Optimal hyperparameters *** ')

        for i in range(0, len(self.opt.x)): 
            print('{}: {}'.format(self.hyperparameters[i], self.opt.x[i]))
            self.values.append(self.opt.x[i])

        dump(opt_method, os.path.join(self.path_to_save, r'optimization_data.pkl'))



    def convergence_plot(self):
        """
        plot convergence plot of the optimization

        args: None
        return: convergence plot
        """
        # Setting up the figure
        fig, ax = plt.subplots(figsize = (8,8))

        fontsize = 16

        plot = plot_convergence((str(self.optimization_method), self.opt))

        plot.legend(loc="best", prop={'size': fontsize}, numpoints=1)
        ax.grid(b = None)
        ax.set_title(' ', fontsize = 18)
        ax.set_xlabel('Number of iterations', fontsize = fontsize)
        ax.set_ylabel('Objective minimum', fontsize = fontsize) 
        ax.tick_params(axis='both', which='major', labelsize=fontsize, direction='in')


        #final_figure
        fig.savefig(os.path.join(self.path_to_save, r'Optimisation_result.png'), dpi = 300, bbox_inches="tight")


    def objective_plot(self):
        """
        Plot objective and corresponding evaluation plots

        args: None
        return: objective and evaluation plots
        """
        _ = plot_objective(self.opt, n_points = 10) 
        _ = plot_evaluations(self.opt)


    def train_model(self):
        """
        Train model with optimal hyperparameters identified 

        args: None
        return: trained model
        """
        # Set model with optimal parameters 
        self.model = self.estimator

        for p, v in zip(self.hyperparameters, self.values):
            self.model.set_params(**{p: v})

        self.model.fit(self.sample_train[self.RFE_features], self.sample_train[self.target].values.ravel())

        return self.model


    def regression_plot(self, X, Y, min_value, max_value):
        """
        Show regression results; this function is recalled using 'evaluate()'

        args: 
        (a) X (type:list); true/observed target values
        (b) Y (type:list); predicted target values
        (c) min_value (type:int); min value to plot i.e. lower limit
        (d) max_value (type:int); max value to plot i.e. upper limit

        return: stats and figure of regression plot
        """
        # Figure
        plt.figure(figsize=(8, 8)) 

        # Predicted vs Actual
        plt.plot(X, Y, 'o', markersize=5, color='black', alpha=0.15)

        # line of best fit
        no_ticks = max_value
        linear_fit = np.linspace(0, no_ticks - 5, no_ticks)
        plt.plot(linear_fit, linear_fit*self.stats_results.params[1] + self.stats_results.params[0], '-', color='tab:blue') 

        # Ideal y=x 
        y = x = np.linspace(0, no_ticks - 5, no_ticks)
        plt.plot(x, y, '--', color='red', alpha=0.8) 

        fontsize = 18
        plt.xlim([min_value, no_ticks])
        plt.ylim([min_value, no_ticks])
        plt.xlabel('True target value', fontsize=fontsize)
        plt.ylabel('Predicted target value', fontsize=fontsize)
        plt.tick_params(axis='both', which='both', labelsize=fontsize, direction="in")
        plt.rcdefaults()

        print('Linear fit has: ')
        print('m = ', self.stats_results.params[1])
        print('c = ', self.stats_results.params[0], '\n')

        plt.savefig(os.path.join(self.path_to_save, r'regression_plot.png'), dpi = 300, bbox_inches="tight")
        plt.show()


    def confusion_matrix(self, target_names):
        """
        Generate confusion matrix plot 

        args: target_names (type:list); list of target classes
        return: conusion matrix plot
        """
        # Pretty confusion matrix 
        disp = plot_confusion_matrix(
                                    self.model, 
                                    self.sample_test[self.RFE_features], 
                                    self.sample_test[self.target],
                                    display_labels=np.array(target_names, dtype='<U10'),
                                    cmap=plt.cm.Blues,
                                    normalize=None
                                    )
        
        fontsize = 13
        plt.tick_params(axis='both', which='major', labelsize=fontsize, direction='in')
        plt.savefig(os.path.join(self.path_to_save, r'Confusion_matrix.png'), dpi = 300, bbox_inches="tight")
        plt.show()


    def evaluate(self, strategy, *args, **kwargs):
        """
        Evaluate the ML model using out-of-sample test set

        args: 
        (a) strategy (type:str); averaging method e.g. 'micro', 'macro', 'weighted'
        (b*) target_names (type:list); list of target classes

        return: stats and plots of result
        """
        if self.problem == 'classification':

            target_names = kwargs.get('target_names')

            # Apply model onto test data
            self.y_test = self.sample_test[self.target]
            self.y_pred =  self.model.predict_proba(self.sample_test[self.RFE_features])
            self.y_pred_2 = self.model.predict(self.sample_test[self.RFE_features])

            # Evaluate metric scores
            print('1. The F-1 score of the model {}\n'.format(f1_score(self.y_test.ravel(), self.y_pred_2, average=strategy)))
            print('2. The recall score of the model {}\n'.format(recall_score(self.y_test.ravel(), self.y_pred_2, average=strategy)))
            print('3. Classification report \n {} \n'.format(classification_report(self.y_test.ravel(), self.y_pred_2, target_names=target_names)))
            print('4. Classification report \n {} \n'.format(multilabel_confusion_matrix(self.y_test.ravel(), self.y_pred_2)))
            print('5. Confusion matrix \n {} \n'.format(confusion_matrix(self.y_test.ravel(), self.y_pred_2)))
            print('6. Accuracy score \n {} \n'.format(accuracy_score(self.y_test.ravel(), self.y_pred_2)))
            print('7. Balanced accuracy score \n {} \n'.format(balanced_accuracy_score(self.y_test.ravel(), self.y_pred_2)))

            # Evaluate matthews correlation coef
            y_test_2 = label_binarize(self.y_test, classes=[i for i in range(self.target_classes)])

            # Convert each row to 1 and 0 based on prob
            all_scores = self.y_pred
            all_scores_2 = np.zeros_like(all_scores)
            all_scores_2[np.arange(len(all_scores)), all_scores.argmax(1)] = 1

            m_corr = list()


            print('8. Matthews corrcoef of Class: ')
            for i in range(self.target_classes):
                corr = matthews_corrcoef(y_test_2[:, i], all_scores_2[:, i])
                m_corr.append(corr)
                print(str(target_names[i]) + ': ', corr)

            print('9. Matthews macro corrcoef \n {} \n'.format(sum(m_corr)/3)) 

            # Get pretty conusion ma trix
            self.confusion_matrix(target_names)


        elif self.problem == 'regression':

            adjusted = kwargs.get('adjusted')
            min_value = kwargs.get('min_value')
            max_value = kwargs.get('max_value')

            # Apply model onto test data
            self.y_test = self.sample_test[self.target]
            self.y_pred = self.model.predict(self.sample_test[self.RFE_features])
            self.id_index = self.sample_test.index.tolist()


            df_pred = pd.DataFrame(
                                    {'task_id': self.id_index, 
                                    str(self.target): self.y_test, 
                                    'pred_target': self.y_pred
                                    })

            # Create a column to eliminate negative values
            df_pred['adjusted_pred_target'] = df_pred['pred_target']
            df_pred['adjusted_pred_target'] = df_pred['adjusted_pred_target'].apply(lambda x: 0 if x < 0 else x)

            X = df_pred[self.target]

            if adjusted == True:
                Y = df_pred['adjusted_pred_target']
            else:
                Y = df_pred['pred_target'] 

            # Stats
            self.stats_results = sm.OLS(Y,sm.add_constant(X)).fit()

            print(self.stats_results.summary())

            print('MAE: ', mean_absolute_error(X, Y))
            print('MSE: ', mean_squared_error(X, Y))
            print('RMSE: ', mean_squared_error(X, Y, squared=False))
            print('R-squared: ', r2_score(X, Y))
            print('Max error: ', max_error(X, Y))
            print('Explained_variance_score: ', explained_variance_score(X, Y, multioutput='variance_weighted'))

            # Plot figure
            self.regression_plot(X, Y, min_value, max_value)


    def ROC(self, overall_performance, *args, **kwargs):
        """
        Generate ROC plot for the classification problem

        args: 
        (a) overall_performance (type:bool); whether to plot the overall average, where strategy determines the method of averaging
        (b*) strategy (type:str); averaging method e.g. 'micro', 'macro', 'weighted'

        return: figure of ROC 
        """
        strategy = kwargs.get('strategy')
        self.y_test = self.sample_test[self.target]
        self.y_pred =  self.model.predict_proba(self.sample_test[self.RFE_features])
        self.y_pred_2 = self.model.predict(self.sample_test[self.RFE_features])

        # Compute ROC curve and ROC area for each class
        self.fpr = dict()
        self.tpr = dict()
        n_classes = self.y_pred.shape[1]
        roc_auc = dict()

        self.y_test_2 = label_binarize(self.y_test, classes = list(range(n_classes)))


        #################### Micro
        for i in range(n_classes):
            self.fpr[i], self.tpr[i], _ = roc_curve(self.y_test_2[:, i], self.y_pred[:, i])
            roc_auc[i] = auc(self.fpr[i], self.tpr[i])

        # Compute micro-average ROC curve and ROC area
        self.fpr["micro"], self.tpr["micro"], _ = roc_curve(self.y_test_2.ravel(), self.y_pred.ravel())
        roc_auc["micro"] = auc(self.fpr["micro"], self.tpr["micro"])

        #################### Macro
        # First aggregate all false positive rates
        all_fpr = np.unique(np.concatenate([self.fpr[i] for i in range(n_classes)]))

        # Then interpolate all ROC curves at this points
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(n_classes):
            mean_tpr += np.interp(all_fpr, self.fpr[i], self.tpr[i])

        # Finally average it and compute AUC
        mean_tpr /= n_classes

        self.fpr["macro"] = all_fpr
        self.tpr["macro"] = mean_tpr
        roc_auc["macro"] = auc(self.fpr["macro"], self.tpr["macro"])

        # Plot all ROC curves
        plt.figure(figsize=(8,8))
        if overall_performance == True:
            if strategy == 'micro':
                plt.plot(
                        self.fpr["micro"], self.tpr["micro"],
                        label='micro-average ROC (AUC = {0:0.3f})'
                            ''.format(roc_auc["micro"]),
                        color='tab:green', linestyle='-', linewidth=4)

            if strategy == 'macro':
                plt.plot(
                        self.fpr["macro"], self.tpr["macro"],
                        label='macro-average ROC (AUC = {0:0.3f})'
                            ''.format(roc_auc["macro"]),
                        color='tab:blue', linestyle='-', linewidth=4)

        if overall_performance == False:
            # Individual class
            lw = 2
            colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
            for i, color in zip(range(n_classes), colors):
                plt.plot(
                        self.fpr[i], self.tpr[i], color=color, lw=lw,
                        label='ROC curve of class {0} (AUC = {1:0.3f})'
                        ''.format(i, roc_auc[i]))

        # Plot curves
        fontsize = 18
        lw=2

        plt.plot([0, 1], [0, 1], 'k--', lw=lw)
        plt.xlim([0.0, 1])
        plt.ylim([0.0, 1.01])
        plt.xlabel('False Positive Rate', fontsize=fontsize)
        plt.ylabel('True Positive Rate', fontsize=fontsize)
        plt.tick_params(axis='both', which='major', labelsize=fontsize)
        plt.legend(loc="lower right", fontsize=fontsize, framealpha=1)

        #final_figure
        plt.savefig(os.path.join(self.path_to_save, r'Receiver_operating_characteristic_curve.png'), dpi = 300, bbox_inches="tight")
        plt.show()

        plt.show()


    def DET(self, strategy):
        """
        Generate DET plot for the classification problem

        args: strategy (type:str); averaging method e.g. 'micro', 'macro', 'weighted'
        return: figure of DET curve
        """
        # Detection Error Trade-off Curve
        fnr_macro = 1 - self.tpr['macro']
        fnr_micro = 1 - self.tpr['micro']

        # Plot curves
        fontsize = 18
        linewidth = 2

        plt.figure(figsize = (8,8))
        if strategy == 'macro':
            plt.plot(
                    fnr_macro, self.fpr['macro'] ,
                    label='macro-average ERR',
                    color='tab:blue', 
                    linestyle='-',
                    linewidth=linewidth)

        if strategy == 'micro':
            plt.plot(
                    fnr_micro, self.fpr['micro'] ,
                    label='micro-average ERR ',
                    color='tab:green', 
                    linestyle='-',
                    linewidth=linewidth)

        lw=2
        plt.plot([0, 1], [0, 1], 'k--', lw=lw)
        plt.xlim([0.0, 1])
        plt.ylim([0.0, 1])
        plt.xlabel('False Negative Rate', fontsize=fontsize)
        plt.ylabel('False Positive Rate', fontsize=fontsize)
        plt.tick_params(axis='both', which='major', labelsize=fontsize, direction='in')
        plt.legend(loc="upper right", fontsize=fontsize, framealpha=1)

        #final_figure
        plt.savefig(os.path.join(self.path_to_save, r'detection_error_tradeoff_curves_v1.png'), dpi = 300, bbox_inches="tight")
    
        plt.show()

    
    def PR(self):
        """
        Generate PR curve for the classification problem

        args: None
        return: figure of PR curve
        """
        self.y_test = self.sample_test[self.target]
        self.y_pred =  self.model.predict_proba(self.sample_test[self.RFE_features])
        #self.y_pred_2 = self.model.predict(self.sample_test[self.RFE_features])

        # For each class
        n_classes = self.y_pred.shape[1]
        precision = dict()
        recall = dict()
        average_precision = dict()
        thresholds = dict()
        
        self.y_test_2 = label_binarize(self.y_test, classes = list(range(n_classes))) 
            
        # For each class / for the top classifier
        for i in range(n_classes):
            precision[i], recall[i], _ = precision_recall_curve(self.y_test_2[:, i], self.y_pred[:, i])
            average_precision[i] = average_precision_score(self.y_test_2[:, i], self.y_pred[:, i])
                
        precision["micro"], recall["micro"], thresholds['micro'] = precision_recall_curve(self.y_test_2.ravel(),self.y_pred.ravel())
            
        average_precision["micro"] = average_precision_score(self.y_test_2, self.y_pred, average="micro")
        average_precision["weighted"] = average_precision_score(self.y_test_2, self.y_pred, average="weighted")
        average_precision["macro"] = average_precision_score(self.y_test_2, self.y_pred, average="macro")
    
        print('Average precision score, micro-averaged over all classes: {0:0.3f}'
                    .format(average_precision["micro"]))

        print('Average precision score, macro-averaged over all classes: {0:0.3f}'
                    .format(average_precision["macro"]))

        print('Average precision score, weighted-averaged over all classes: {0:0.3f}'
                    .format(average_precision["weighted"]))

        #print('PR_AUC_micro: ', auc(recall["micro"], precision["micro"]))


        # Plot figure
        plt.figure(figsize = (8,8))

        fontsize = 18

        plt.step(
                recall['micro'], precision['micro'], 
                where='post', 
                lw=2, 
                color='tab:blue', 
                label='Micro-averaged PR (AP = 0.995)'
                )

        labelsize = 18

        plt.xlabel('Recall',fontsize=fontsize)
        plt.ylabel('Precision',fontsize=fontsize)
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.tick_params(axis='both', which='major', labelsize=labelsize, direction='in')
        plt.legend(fontsize=fontsize, loc="lower left", framealpha=1.0)

        f_scores = np.linspace(0.2, 0.8, num=4)
        lines = []
        labels = []

        n = 0
        for f_score in f_scores:
            x = np.linspace(0.001, 1.0)
            y = f_score * x / (2 * x - f_score)
            l, = plt.plot(x[y >= 0], y[y >= 0], color='gray', alpha=0.2)
            
            #plt.annotate('f1={0:0.1f}'.format(f_score), xy=(0.9, y[45] + 0.02))

        # Location of the annotation
        x0 = [0.13, 0.26, 0.43, 0.67]
        y0 = [0.2, 0.4, 0.6, 0.8]
        n = 0
        fontsize2 = 14

        while n < len(x0):
            if n < 0:
                plt.annotate('F1=' + str(y0[n]), xy=(x0[n], 0.99 + 0.02),fontsize=fontsize2)
            else:
                plt.annotate('F1=' + str(y0[n]), xy=(x0[n], 0.99 + 0.02),fontsize=fontsize2)
            n = n + 1 

        #Save figure
        plt.savefig(os.path.join(self.path_to_save, r'precision_recall.png'), dpi = 300, bbox_inches="tight")
        
        plt.show()

