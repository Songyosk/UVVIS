"""
Gradient boosting feature selection with preliminary scan of the hyperparameter space using the gride search method

Author: Son Gyo Jung
Email: sgj13@cam.ac.uk
"""

import os 
import numpy as np
import pandas as pd
import joblib
import random
import seaborn as sns
import matplotlib.pyplot as plt

from copy import deepcopy
from time import time
from itertools import product

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, label_binarize
from sklearn import metrics 
from sklearn.metrics import multilabel_confusion_matrix, roc_curve, roc_auc_score, max_error, \
                            auc, f1_score, classification_report, recall_score, precision_recall_curve, \
                            balanced_accuracy_score, confusion_matrix, accuracy_score, average_precision_score, \
                            hamming_loss, matthews_corrcoef, mean_squared_error, mean_absolute_error, r2_score

from imblearn.over_sampling import (RandomOverSampler, 
                                    SMOTE,
                                    SMOTENC,
                                    BorderlineSMOTE,
                                    ADASYN)

from lightgbm.sklearn import LGBMClassifier, LGBMRegressor
from xgboost import XGBClassifier, XGBRegressor


class GBFS():
    """
    Class used to select preliminary subset of features that maximizes the choice performance metric

    args: 
        (1) path_to_file (type:str) - location of the training set
        (2) path_to_save (type:str) - location to save new data files
        (3) oversampled_it (type:bool) - whether to oversampled the training data; choose False if already oversampled
        (4) problem (type:str) - whether it is a 'classification' or 'regression' problem
        (5*) target_classes (type:int) - for classification, specify the number of target classes

    return: 
        (1) list of features selected during GBFS
    """

    def __init__(self, path_to_file, path_to_save, oversampled_it, problem, *args, **kwargs):
        self.path_to_save = path_to_save
        self.sample_train = joblib.load(path_to_file) 

        # Last column taken as the target variable or classes
        first_feature_col = kwargs.get('first_feature_col')

        if first_feature_col is not None:
            self.features = self.sample_train.columns.values[first_feature_col:-1]
        else:
            self.features = self.sample_train.columns.values[0:-1]

        self.target = self.sample_train.columns.values[-1]

        print('Name of target column: ', self.target)
        print('No. of exploratory features: ', len(self.features) )

        self.oversampled_it = oversampled_it
        self.problem = problem
        self.target_classes = kwargs.get('target_classes')



    def oversample(self, df, technique, *args, **kwargs):
        """
        Oversample with various technique: 

        (a) 'ros'
        (b)'smoothed_ros'
        (c)'smote'
        (d)'smote_nc'
        (e)'smote_borderline1'
        (f)'smote_borderline2'
        (g)'adasyn'

        This function is embedded into the 'grid_search()' function

        args:
            (1) df (pandas.Dataframe) - training data
            (2) technique (type:str) - oversampling technique to use
            (b*) categorical_features (type:list); list of indices specifying the position of categorical columns; this is only applicable when using 'smote_nc'
        
        return: 
            (1) pandas.Dataframe with oversampled data
        """

        #Oversample the training set
        x = df[self.features].values
        y = df[self.target].values


        #Different oversampling techniques 
        if technique == 'ros':
            os = RandomOverSampler()

        elif technique == 'smoothed_ros':
            os = RandomOverSampler(shrinkage=0.15)

        elif technique == 'smote':
            os = SMOTE()

        elif technique == 'smote_nc':
            self.categorical_features = kwargs.get('categorical_features')
            os = SMOTENC(categorical_features=categorical_features, k_neighbors=5)

        elif technique == 'smote_borderline1':
            os = BorderlineSMOTE(k_neighbors=3, m_neighbors=15, kind='borderline-1')

        elif technique == 'smote_borderline2':
            os = BorderlineSMOTE(k_neighbors=3, m_neighbors=15, kind='borderline-2')

        elif technique == 'adasyn':
            os = ADASYN()


        # Fit on data
        x_oversampled, y_oversampled = os.fit_resample(x, y)

        # Create pandas.Dataframe
        oversampled_train = pd.concat([pd.DataFrame(data=x_oversampled), pd.DataFrame(data=y_oversampled, columns=[self.target])], axis=1)

        # Add column names
        oversampled_train.columns = df.columns 

        print('   No. of rows in training set after oversampling:', len(oversampled_train))


        return oversampled_train



    def grid_search(self, model, params, stratify, cv_folds, oversample_technique, *args, **kwargs):
        """
        Perform grid search to conduct a preliminary search of the hyperparameter space
        This function takes either raw training data or oversampled training data (as specified during initialization i.e. 'oversample_it'),
        
        Note 20% of training set is used as out-of-sample validation set

        Oversample with various technique: 
        (a) 'ros'
        (b)'smoothed_ros'
        (c)'smote'
        (d)'smote_nc'
        (e)'smote_borderline1'
        (f)'smote_borderline2'
        (g)'adasyn'

        args:
            (1) model (sklearn.estimator) - the model to be optimised
            (2) params (type:int or float) - hyperparameter values
            (3) stratify (type:bool) - whether to stratify data while splitting into training and validation sets
            (4) cv_folds (type:int) - number of cross validation
            (4) oversample_technique (type:str) - oversample method to employ

        Returns: 
            (1) model fitted with the optimal hyperparameters 
        """

        # Define the lowest score
        if self.problem == 'classification':
            max_score = 0

        elif self.problem == 'regression':
            max_score = float('-inf')


        #Permutations based on the values of the hyperparameters
        params_perm = list(product(*params.values())) 

        print('Total no. of permutations:', len(params_perm))


        for i, chosen_param in enumerate(params_perm):
            print('\n')
            print('   (' + str(i+1) + ' of ' + str(len(params_perm)) + ')', ' Attempt: ', list(zip(params.keys(), chosen_param)))
            

            metric_score = []

            #Set the parameters for the chosen estimator/model 
            for p, v in zip(params.keys(), chosen_param):
                model.set_params(**{p: v})

                
            #Oversample data and train the model. Compute mean performance metric using out-of-sample validation set and the chosen CV fold
            for fold in range(cv_folds):
                #Each fold will adjust the random_state
                if stratify == True:
                    sample_tr, sample_va = train_test_split(
                                                            self.sample_train, 
                                                            test_size = 0.2, 
                                                            random_state = fold + random.randint(0, 100), 
                                                            stratify = self.sample_train[self.target].to_list()
                                                            )
                
                elif stratify == False:
                    sample_tr, sample_va = train_test_split(
                                                            self.sample_train, 
                                                            test_size = 0.2, 
                                                            random_state = fold + random.randint(0, 100)
                                                            )

                print('   No. of rows in the training set:', len(sample_tr))


                if self.problem == 'classification':
                    if self.oversampled_it == True:
                        print('   Oversampling training data...')

                        # Oversample data
                        oversampled_tr = self.oversample(
                                                        df = sample_tr, 
                                                        technique = oversample_technique
                                                        )


                        # Scale features
                        scaling = MinMaxScaler(feature_range=(0, 1)) #Range can be adjusted

                        sample_tr_features = pd.DataFrame(
                                                        scaling.fit_transform(oversampled_tr[self.features].values), 
                                                        columns=oversampled_tr[self.features].columns, 
                                                        index=oversampled_tr[self.features].index
                                                        )

                        sample_va_features = pd.DataFrame(
                                                        scaling.fit_transform(sample_va[self.features].values), 
                                                        columns=sample_va[self.features].columns, 
                                                        index=sample_va[self.features].index
                                                        )

                        oversampled_tr = pd.concat([sample_tr_features, oversampled_tr[self.target]], axis=1)
                        sample_va = pd.concat([sample_va_features, sample_va[self.target]], axis=1)


                        # Fit to model
                        model.fit(oversampled_tr[self.features], oversampled_tr[self.target].values.ravel())


                    elif self.oversampled_it == False:
                        # Scale features
                        scaling = MinMaxScaler(feature_range=(0, 1)) # Range can be adjusted

                        sample_tr_features = pd.DataFrame(
                                                        scaling.fit_transform(sample_tr[self.features].values), 
                                                        columns=sample_tr[self.features].columns, 
                                                        index=sample_tr[self.features].index
                                                        )

                        sample_va_features = pd.DataFrame(
                                                        scaling.fit_transform(sample_va[self.features].values), 
                                                        columns=sample_va[self.features].columns, 
                                                        index=sample_va[self.features].index
                                                        )

                        sample_tr = pd.concat([sample_tr_features, sample_tr[self.target]], axis=1)
                        sample_va = pd.concat([sample_va_features, sample_va[self.target]], axis=1)


                        # Fit to model
                        model.fit(sample_tr[self.features], sample_tr[self.target].values.ravel())
                    
                    try:
                        score = roc_auc_score(
                                            sample_va[self.target], 
                                            model.predict_proba(sample_va[self.features]),
                                            average='weighted', 
                                            multi_class="ovr"
                                            )

                        metric_score += [score]

                    except:
                        score = roc_auc_score(
                                            sample_va[self.target], 
                                            model.predict(sample_va[self.features]),
                                            average='weighted', 
                                            multi_class="ovr"
                                            )

                        metric_score += [score]


                elif self.problem == 'regression':
                    #Scale features
                    scaling = MinMaxScaler(feature_range=(0, 1)) #Range can be adjusted

                    sample_tr_features = pd.DataFrame(
                                                    scaling.fit_transform(sample_tr[self.features].values), 
                                                    columns=sample_tr[self.features].columns, 
                                                    index=sample_tr[self.features].index
                                                    )

                    sample_va_features = pd.DataFrame(
                                                    scaling.fit_transform(sample_va[self.features].values), 
                                                    columns=sample_va[self.features].columns, 
                                                    index=sample_va[self.features].index
                                                    )

                    sample_tr = pd.concat([sample_tr_features, sample_tr[self.target]], axis=1)
                    sample_va = pd.concat([sample_va_features, sample_va[self.target]], axis=1)


                    #Fit to model
                    model.fit(sample_tr[self.features], sample_tr[self.target].values.ravel())

                    try:
                        score = -1 * mean_squared_error(
                                                        sample_va[self.target], 
                                                        model.predict(sample_va[self.features])
                                                        )
                        metric_score += [score]
                    
                    except:
                        print('Error; current attempt skipped')

                        pass


            mean_score = np.mean(metric_score)

            print('   Metric score: ', '%.5f' % mean_score, flush=True)


            #Update hyperparameters
            if mean_score > max_score:
                print('   [*** Current optimal Hyperparameters ***]')

                max_score = mean_score

                self.best_model = deepcopy(model)


        return self.best_model



    def run(self, boosting_method):
        """
        Execute the grid search using the selected boosting method
        Other parameters should be adjusted inside this function before execution

        Note: 
        For classification, multi-class models are defined as shown below
        This can be changed into a binary problem by changing the 'objective' to 'binary' for LGBMClassifier, or to 'binary:logistic' or 'binary:logitraw' for XGBClassifier (see description in links below)
        
        https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMClassifier.html
        https://xgboost.readthedocs.io/en/latest/parameter.html
        https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMRegressor.html

        args: 
            (1) boosting_method (type:str) - either 'lightGBM' or 'XGBoost'

        return: 
            (2) model fitted with the optimal parameters 
        """
        methods = [
                    {'name': 'lightGBM', #https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMClassifier.html
                    'type': 'classification',
                    'estimator': LGBMClassifier(
                                                boosting_type = 'gbdt',
                                                objective = 'multiclass',
                                                importance_type = 'gain'
                                                ),                 

                    'hyperparameters': {
                                        'n_estimators':[50, 100, 300, 600],
                                        'learning_rate':[0.1, 0.2],
                                        'num_leaves':[30, 40, 50, 60]
                                        }
                    },


                    {'name': 'XGBoost', #https://xgboost.readthedocs.io/en/latest/parameter.html
                    'type': 'classification',
                    'estimator': XGBClassifier(
                                                objective = 'multi:softprob',
                                                booster = 'gbtree',
                                                importance_type = 'total_gain'
                                                ),
                        
                    'hyperparameters': {
                                        'n_estimators':[50, 100, 300, 600],
                                        'learning_rate':[0.1, 0.2],
                                        'num_leaves':[30, 40, 50, 60]
                                        }
                    },


                    {'name': 'lightGBM', #https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMRegressor.html
                    'type': 'regression',
                    'estimator': LGBMRegressor(
                                                boosting_type = 'gbdt',
                                                objective = 'regression',
                                                importance_type = 'gain'
                                                ),

                    'hyperparameters': {
                                        'n_estimators':[50, 100, 300, 600],
                                        'learning_rate':[0.1, 0.2],
                                        'num_leaves':[30, 40, 50, 60]
                                        }
                    },


                    {'name': 'XGBoost', #https://xgboost.readthedocs.io/en/stable/parameter.html
                    'type': 'regression',
                    'estimator': XGBClassifier(
                                                objective = 'reg:squarederror',
                                                booster = 'gbtree',
                                                random_state = 42,
                                                importance_type = 'total_gain'
                                                ),
                        
                    'hyperparameters': {
                                        'n_estimators':[50, 100, 300, 600],
                                        'learning_rate':[0.1, 0.2],
                                        'num_leaves':[30, 40, 50, 60]
                                        }
                    }]


        #Store model hyperparameters
        models = dict()


        for method in methods:
            if self.problem == 'classification' and method['type'] == self.problem:
                if method['name'] == boosting_method:

                    print('Model employed: ', method['name'])

                    models[boosting_method] = self.grid_search(
                                                            problem = 'classification',
                                                            model = method['estimator'],
                                                            params = method['hyperparameters'],
                                                            stratify = True,
                                                            cv_folds = 5,
                                                            oversample_technique = 'smoothed_ros'
                                                            )


            elif self.problem == 'regression' and method['type'] == self.problem:
                if method['name'] == boosting_method:

                    print('Model employed: ', method['name'])

                    models[boosting_method] = self.grid_search(
                                                            problem = 'regression',
                                                            model = method['estimator'],
                                                            params = method['hyperparameters'],
                                                            stratify = False,
                                                            cv_folds = 5,
                                                            oversample_technique = None
                                                            )


        #save models 
        self.saved_model = models[boosting_method]

        joblib.dump(self.saved_model, os.path.join(self.path_to_save, r'saved_model_from_GBFS.pkl'))
        
        print('\n', 'Model saved as "saved_model_from_GBFS.pkl"')


        return self.saved_model



    def feature_relevance(self, plot, no_of_features):
        """
        Obtain the feature relevance score

        args:
            (1) plot (type:bool) - whether to generate feature ranking plot
            (2) no_of_features (type:int) - number of features to plot, starting from the most relevant feature

        return: 
            (1) pandas.Dataframe of feature relevance score (pkl)
        """

        model = self.saved_model 

        self.feature_score = pd.DataFrame({'feature': self.features, 'relevance_score': model.feature_importances_})
        self.feature_score = self.feature_score.sort_values(by = 'relevance_score', ascending = False)
        self.feature_score = self.feature_score.reset_index(drop = True)


        #Save data
        joblib.dump(self.feature_score, os.path.join(self.path_to_save, r'feature_relevance_score.pkl'))

        print('Result saved as: "feature_relevance_score.pkl"')


        if plot == True:
            #Plot data
            sns.set(rc = {'figure.figsize':(10, 10)})

            fig = sns.barplot(x = 'relevance_score', y = 'feature', data = self.feature_score[:no_of_features])
            fig.set(xlabel = 'Relevance score', ylabel = 'Feature')

            plt.savefig(os.path.join(self.path_to_save, r'feature_relevance_plot.png'), dpi = 300)

            print('Figure saved as: "feature_relevance_plot.png"')


        return self.feature_score



    def recursive_selection(self, stratify, oversample_technique, chosen_metric, average, no_to_terminate, max_no_imp):
        """
        Find subset of features that maxmises the chosen performance metric

        Oversample with various technique: 
        (a) 'ros'
        (b)'smoothed_ros'
        (c)'smote'
        (d)'smote_nc'
        (e)'smote_borderline1'
        (f)'smote_borderline2'
        (g)'adasyn'

        Args: 
            (1) stratify (type:bool) - whether to stratify the dataset while splitting
            (2) oversample_technique (type:str) - oversampling technique to use
            (3) chosen_metric (type:str) - the metric used for the convergence criterion, where the name convention is consistent with scikit-learn
            (4) average (type:str) - averaging method for calculating the metrics, i.e. 'micro', 'macro', 'weighted'
            (5) no_to_terminate (type:int) - maximum number of features to consider given convergence criterion is not met
            (6) max_no_imp (type:int) - maximum number of no improvements before terminating

        Returns: 
            (1) subset of features that maximises the chosen metric for the given target 
        """

        # Define the criteria
        no_to_terminate = no_to_terminate
        max_no_imp = max_no_imp

        # Define range
        start_no = 1
        n_range = [i for i in range(start_no, len(self.feature_score['feature']) + 1)]

        # Define lists to append results
        if self.problem == 'classification':
            tr_b_ac, tr_ac, tr_hl, tr_pr, tr_roc, tr_f1 = list(), list(), list(), list(), list(), list()
            va_b_ac, va_ac, va_hl, va_pr, va_roc, va_f1 = list(), list(), list(), list(), list(), list()

        elif self.problem == 'regression':
            tr_mae, tr_rmse, tr_mse, tr_r_sq, tr_error = list(), list(), list(), list(), list()
            va_mae, va_rmse, va_mse, va_r_sq, va_error = list(), list(), list(), list(), list()


        #Split data
        if self.problem == 'classification' and stratify == True:
            sample_tr, sample_va = train_test_split(self.sample_train, test_size = 0.2, stratify = self.sample_train[self.target].to_list())

        else:
            sample_tr, sample_va = train_test_split(self.sample_train, test_size=0.2)

        print('   No. of rows in training set: ', len(sample_tr))


        # Oversample for classification
        if self.oversampled_it == True:
            oversampled_tr = self.oversample(df = sample_tr, technique = oversample_technique)


            # Scale features
            scaling = MinMaxScaler(feature_range = (0, 1)) #Range can be adjusted

            sample_tr_features = pd.DataFrame(
                                            scaling.fit_transform(oversampled_tr[self.features].values), 
                                            columns = oversampled_tr[self.features].columns, 
                                            index = oversampled_tr[self.features].index
                                            )

            sample_va_features = pd.DataFrame(
                                            scaling.fit_transform(sample_va[self.features].values), 
                                            columns = sample_va[self.features].columns, 
                                            index = sample_va[self.features].index
                                            )

            oversampled_tr = pd.concat([sample_tr_features, oversampled_tr[self.target]], axis = 1)
            sample_va = pd.concat([sample_va_features, sample_va[self.target]], axis = 1)


        elif self.oversampled_it == False:

            # Scale features
            scaling = MinMaxScaler(feature_range = (0, 1)) #Range can be adjusted

            sample_tr_features = pd.DataFrame(
                                            scaling.fit_transform(sample_tr[self.features].values), 
                                            columns = sample_tr[self.features].columns, 
                                            index = sample_tr[self.features].index
                                            )

            sample_va_features = pd.DataFrame(
                                            scaling.fit_transform(sample_va[self.features].values), 
                                            columns = sample_va[self.features].columns, 
                                            index = sample_va[self.features].index
                                            )

            sample_tr = pd.concat([sample_tr_features, sample_tr[self.target]], axis = 1)
            sample_va = pd.concat([sample_va_features, sample_va[self.target]], axis = 1)


        no_imp = 0

        for n in n_range:
            start_time = time()
            selected = self.feature_score['feature'][:n].tolist()
            model = self.saved_model 

            model.fit(sample_tr[selected], sample_tr[self.target].values.ravel())
            
            if self.problem == 'classification':
                ## Training set
                y_train = sample_tr[self.target]
                y_train_pred = model.predict_proba(sample_tr[selected])
                y_train_pred_2 = model.predict(sample_tr[selected])

                y_train_bin = label_binarize(y_train, classes=[i for i in range(self.target_classes)])  
                
                ac_train = accuracy_score(y_train, y_train_pred_2)
                b_ac_train = balanced_accuracy_score(y_train, y_train_pred_2)
                hl_train = hamming_loss(y_train, y_train_pred_2)
                train_f1_score = f1_score(y_train, y_train_pred_2, average = average)
                train_roc_score = roc_auc_score(y_train, y_train_pred, average = average, multi_class = 'ovr')
                train_avg_p_score = average_precision_score(y_train_bin, y_train_pred, average = average)

                tr_b_ac.append(b_ac_train)
                tr_ac.append(ac_train)
                tr_hl.append(hl_train)
                tr_roc.append(train_roc_score)
                tr_pr.append(train_avg_p_score)
                tr_f1.append(train_f1_score) 
                

                ## Validation set   
                y_va = sample_va[self.target]
                y_va_pred = model.predict_proba(sample_va[selected])
                y_va_pred_2 = model.predict(sample_va[selected])

                y_va_bin = label_binarize(y_va, classes=[i for i in range(self.target_classes)]) 

                ac_va = accuracy_score(y_va, y_va_pred_2)
                b_ac_va = balanced_accuracy_score(y_va, y_va_pred_2)
                hl_va = hamming_loss(y_va, y_va_pred_2)
                va_f1_score = f1_score(y_va, y_va_pred_2, average = average)
                va_roc_score = roc_auc_score(y_va, y_va_pred,  average = average, multi_class = 'ovr')
                va_avg_p_score= average_precision_score(y_va_bin, y_va_pred, average = average)
                        
                va_b_ac.append(b_ac_va)
                va_ac.append(ac_va)
                va_hl.append(hl_va)
                va_roc.append(va_roc_score)
                va_pr.append(va_avg_p_score)
                va_f1.append(va_f1_score)        
                

                # Print results for each iteration
                print('No. of features considering: ', len(selected))
                # print('Features trying: ', selected)
                print('')
                print('n=%d: train_acc=%.4f, validation_acc=%.4f \n' % (n, ac_train, ac_va))
                print('n=%d: train_b_acc=%.4f, validation_b_acc=%.4f \n' % (n, b_ac_train, b_ac_va))
                print('n=%d: train_hl=%.4f validation_hl=%.4f \n' % (n, hl_train, hl_va))
                print('n=%d: train_f1=%.4f, validation_f1=%.4f \n' % (n, train_f1_score, va_f1_score))
                print('n=%d: train_roc_auc=%.4f, validation_roc_auc=%.4f \n' % (n, train_roc_score, va_roc_score))
                print('n=%d: train_avg_precision)=%.4f, validation_avg_precision)=%.4f \n' % (n, train_avg_p_score, va_avg_p_score))

                # Print time taken
                print("--- %s seconds ---" % (time() - start_time), '\n')


                if n > 2:
                    if chosen_metric == 'f1_score':
                        metric = va_f1

                    elif chosen_metric == 'accuracy':
                        metric = va_ac

                    elif chosen_metric == 'balanced_accuracy':
                        metric = va_b_ac

                    elif chosen_metric == 'hamming_loss':
                        metric = va_hl

                    elif chosen_metric == 'roc_auc':
                        metric = va_roc

                    elif chosen_metric == 'average_precision':
                        metric = va_pr

                    if round(metric[-2], 3) >= round(metric[-1], 3):
                        no_imp = no_imp + 1

                    else:
                        no_imp = 0

                    print('No. of no improvements: ', no_imp)


                if no_imp == max_no_imp:
                    print('Terminated: no improvements for ' + str(no_imp) + ' iterations')

                    break


                if n == no_to_terminate:
                    print('Terminated: max. no. of iterations reached')

                    break


            if self.problem == 'regression':

                ## Training set
                y_train = sample_tr[self.target]
                y_train_pred = model.predict(sample_tr[selected]) 

                mae_train = mean_absolute_error(y_train, y_train_pred)
                mse_train = mean_squared_error(y_train, y_train_pred)
                rmse_train = mean_squared_error(y_train, y_train_pred, squared=False)
                r2_train = r2_score(y_train, y_train_pred)
                err_train = max_error(y_train, y_train_pred)

                tr_mae.append(mae_train)
                tr_mse.append(mse_train)
                tr_rmse.append(rmse_train) 
                tr_r_sq.append(r2_train) 
                tr_error.append(err_train)              


                ## Validation set
                y_va =  sample_va[self.target]
                y_va_pred =  model.predict(sample_va[selected])

                mae_va = mean_absolute_error(y_va, y_va_pred)
                mse_va = mean_squared_error(y_va, y_va_pred)
                rmse_va = mean_squared_error(y_va, y_va_pred, squared=False)
                r2_va = r2_score(y_va, y_va_pred)
                err_va = max_error(y_va, y_va_pred)
                
                va_mae.append(mae_va)
                va_mse.append(mse_va)
                va_rmse.append(rmse_va) 
                va_r_sq.append(r2_va) 
                va_error.append(err_va) 


                print('No. of features considering: ', len(selected))
                # print('Features trying: ', selected)
                print('')
                print('n=%d: mae_train=%.4f, mae_validation=%.4f ' % (n, mae_train, mae_va))
                print('n=%d: mse_train=%.4f, mse_validation=%.4f ' % (n, mse_train, mse_va))
                print('n=%d: rmse_train=%.4f, rmse_validation=%.4f ' % (n, rmse_train, rmse_va))
                print('n=%d: r2_train=%.4f, r2_validation=%.4f ' % (n, r2_train, r2_va))
                print('n=%d: max_error_train=%.4f, max_error_validation=%.4f ' % (n, err_train, err_va))
            
                print("--- %s seconds ---" % (time() - start_time), '\n')


                if n > 2:
                    if chosen_metric == 'mae':
                        metric = va_mae

                    elif chosen_metric == 'rmse':
                        metric = va_rmse

                    elif chosen_metric == 'r2':
                        metric = va_r_sq

                    if round(metric[-2], 3) >= round(metric[-1], 3):
                        no_imp = no_imp + 1

                    else:
                        no_imp = 0

                    print('No. of no improvements: ', no_imp)


                if no_imp == max_no_imp:
                    print('Terminated: no improvements for ' + str(no_imp) + ' iterations')

                    break

                if n == no_to_terminate:
                    print('Terminated: max. no. of iterations reached')

                    break

                
            if self.problem == 'classification':
                self.result = pd.DataFrame(data = list(zip(
                                                            n_range, 
                                                            tr_ac, va_ac,
                                                            tr_b_ac, va_b_ac,
                                                            tr_hl, va_hl,
                                                            tr_pr, va_pr,
                                                            tr_roc, va_roc, 
                                                            tr_f1, va_f1
                                                            )))

                self.result.columns = [
                                        'no_of_features', 
                                        'train_acc', 'validation_acc',
                                        'train_b_acc', 'validation_b_acc', 
                                        'train_hamming', 'validation_hamming', 
                                        'train_avg_precision', 'validation_avg_precision', 
                                        'train_roc_auc', 'validation_roc_auc', 
                                        'train_f1', 'validation_f1'
                                        ]


            elif self.problem == 'regression':
                self.result = pd.DataFrame(data = list(zip(
                                                            n_range, 
                                                            tr_mae, va_mae, 
                                                            tr_mse, va_mse, 
                                                            tr_rmse, va_rmse, 
                                                            tr_r_sq, va_r_sq, 
                                                            tr_error, va_error
                                                            )))

                self.result.columns = [
                                        'no_of_features', 
                                        'train_mae', 'va_mae', 
                                        'train_mse', 'va_mse', 
                                        'train_rmse', 'va_rmse', 
                                        'train_r_sq', 'va_r_sq', 
                                        'train_max_error', 'va_max_error'
                                        ]


            self.result = self.result.set_index('no_of_features')

        joblib.dump(self.result, os.path.join(self.path_to_save, r'GBFS_result.pkl'))

        print('Result saved as: "GBFS_result.pkl"')


        return self.result



    def convergence_plot(self, *args, **kwargs):
        """
        Generate convergence plot based on the GBFS feature ranking

        args: 
            (1*) train_metric (type:str or list) - metric(s) to plot for training set
            (2*) validation_metric (type:str or list) - metric(s) to plot for training set
        
        return: 
            (1) figure of the convergence plot
        """

        train_metric = kwargs.get('train_metric')
        validation_metric = kwargs.get('validation_metric')


        x = self.result.index.tolist()

        if train_metric != None and validation_metric != None:
            plt.plot(x, self.result[train_metric], '-', label=train_metric)
            plt.plot(x, self.result[validation_metric], '-', label=validation_metric)

        else:
            for i in self.result.columns.values:
                plt.plot(x, self.result[i], '-', label=str(i))
        

        plt.xlabel("Number of features")
        plt.ylabel("Performance metric score")
        plt.legend()
        plt.show()

        plt.savefig(os.path.join(self.path_to_save, r'GBFS_convergence_plot.png'), dpi = 300)

        print('Result saved as: "GBFS_convergence_plot.pkl"')

