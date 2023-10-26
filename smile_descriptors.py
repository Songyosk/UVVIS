"""
Module to generate features using SMILE descriptors
Author: Son Gyo Jung
Email: sgj13@cam.ac.uk
"""  

import numpy as np
import pandas as pd
from pathlib import Path
import joblib
import os
import re
import logging

import deepchem as dc

from rdkit import Chem, DataStructs
from rdkit.Chem import DataStructs
from rdkit.Chem import MACCSkeys
from rdkit.Chem.rdMolDescriptors import CalcMolFormula

from descriptastorus.descriptors import rdDescriptors, rdNormalizedDescriptors
from descriptastorus.descriptors.DescriptorGenerator import MakeGenerator
from descriptastorus.descriptors import rdNormalizedDescriptors




class descriptors():
    """
    Class to achieve multicollinearity reduction
    args: 
        (1) path_to_file (type:str) - location of the data file with features
        (2) path_to_save (type:str) - location to save new data files
        (3) feature_score (type:str) - location of 'feature_relevance_score.pkl'
        (4) no_features (type:int) - number of features to consider starting from the most relevant feature
    return: 
        (1) pandas.Dataframe with collinear features removed
    """

    def __init__(self, path_to_file, path_to_save):
        
        self.path_to_save = path_to_save

        try:
            
            self.df = pd.read_csv(path_to_file)
            
        except:
            
            self.df = joblib.load(path_to_file)


        print('Column names: ', self.df.columns)
        
        print('Number of samples/rows: ', len(self.df.index))
        


    def movecol(self, df, cols_to_move=[], ref_col='', place='After'):
        
        cols = df.columns.tolist()
        
        if place == 'After':
            
            seg1 = cols[:list(cols).index(ref_col) + 1]
            
            seg2 = cols_to_move
            
        if place == 'Before':
            
            seg1 = cols[:list(cols).index(ref_col)]
            
            seg2 = cols_to_move + [ref_col]
            
        
        seg1 = [i for i in seg1 if i not in seg2]
        
        seg3 = [i for i in cols if i not in seg1 + seg2]
        
        
        return(df[seg1 + seg2 + seg3])



    def descriptastorus_features(self, smiles: str):

        # make the normalized descriptor generator
        generator = MakeGenerator((
                                'atompaircounts', 
                                'morgan3counts', 
                                'morganchiral3counts', 
                                'morganfeature3counts', 
                                'rdkit2d', 
                                'rdkit2dnormalized', 
                                'rdkitfpbits'
                                ))
        
        #generator.columns 
        

        # Generate features
        results = generator.process(smiles)

        processed, features = results[0], results[1:]


        if processed is None:
            
            logging.warning("Unable to process smiles %s", smiles)


        return features



    def featurization(self, name, df_raw, col=0):
        
        if name == 'rdkit':
            
            featurizer = dc.feat.RDKitDescriptors()
            
        elif name == 'pubchem':
            
            featurizer = dc.feat.PubChemFingerprint()
            
        elif name == 'elem_net':
            
            featurizer = dc.feat.ElemNetFeaturizer()
            
        elif name == 'elem_prop':
            
            featurizer = dc.feat.ElementPropertyFingerprint()
        
        
        successful_id, failed_id, features = list(), list(), list()
        

        if name == 'rdkit' or name == 'pubchem':
            
            for i in df_raw.index:
                
                try:
                    
                    smi = str(df_raw.iloc[i].tolist()[col])

                    f = featurizer.featurize(smi)

                    successful_id.append(i)
                    
                    features.append(f[0].tolist())
                    
                    print('Featurized: ', i, ' ', smi)

                except:
                    
                    failed_id.append(i)
                    
                    print('Error with: ', i)
                    

        elif name == 'elem_net' or name == 'elem_prop':
            
            for i in df_raw.index:
                
                try:
                    
                    smi = str(df_raw.iloc[i].tolist()[col])

                    mol = Chem.MolFromSmiles(smi)
                    
                    comp = CalcMolFormula(mol)
                    
                    # Remove special characters
                    comp = ''.join(e for e in comp if e.isalnum())
                    
                    print(comp)

                    f = featurizer.featurize([comp])
                    
                    if f[0].size > 0:
                        
                        successful_id.append(i)
                        
                        features.append(f[0].tolist())
                
                        print('Featurized: ', i, ' ', smi)
                        
                    else:
                        failed_id.append(i)
                        
                        print('Error with: ', i)
                
                except:
                    failed_id.append(i)
                    
                    print('Error with: ', i)
                    
                
        elif name == 'maccskeys':
            
            for i in df_raw.index:
                
                try:
                    
                    smi = str(df_raw.iloc[i].tolist()[col])
                    
                    mol = Chem.MolFromSmiles(smi)
                    
                    fp = MACCSkeys.GenMACCSKeys(mol)

                    fp_arr = np.zeros((0,), dtype=int)
                    
                    DataStructs.ConvertToNumpyArray(fp, fp_arr)

                    successful_id.append(i)
                    
                    features.append(fp_arr.tolist())
                    
                    print('Featurized: ', i, ' ', smi)

                except:
                    
                    failed_id.append(i)
                    
                    print('Error with: ', i)


        elif name == 'descriptastorus':
            
            for i in df_raw.index:
                
                try:
                    
                    smi = str(df_raw.iloc[i].tolist()[col])

                    fp = self.descriptastorus_features(smi)
                    
                    successful_id.append(i)
                    
                    features.append(fp)
                    
                    print('Featurized: ', i, ' ', smi)
                    
                except:
                    
                    failed_id.append(i)
                    
                    print('Error with: ', i)

        
        return successful_id, failed_id, features



    def generate(self, name, treat=True, col=0):
        '''
        name = 'rdkit' # rdkit, pubchem, elem_net, elem_prop, maccskeys, descriptastorus
        '''
        
        self.name = name

        successful_id, failed_id, features = self.featurization(self.name, df_raw=self.df, col=col)

        print('Number of successful compounds:', len(successful_id))
        print('Number of unsuccessful compounds:', len(failed_id))
        print('Number of features:', len(features))
        

        col = range(0, len(features[0]))
        
        col = [str(self.name) + '_' + str(i) for i in col]
        

        # Array into pandas
        df_temp = pd.DataFrame(features, index=successful_id, columns=col) 
        
        df_temp.index.name = 'ID'


        self.df_final = pd.merge(df_temp, self.df, left_index=True, right_index=True, how='left')

        self.df_final = self.movecol(self.df_final, cols_to_move=['smiles'], ref_col=str(self.name)+'_0', place='Before')

        self.df_final.index.name = 'ID'


        if treat == True:
            # Replace Nan with zero
            self.df_final = self.df_final.fillna(0)

            # Replace infinity with zero
            self.df_final.replace([np.inf, -np.inf], 0, inplace=True)


        print('Failed ID:')
        
        
        return failed_id



    def save(self, file_name, csv=True):
        
        # Save data
        joblib.dump(self.df_final, os.path.join(self.path_to_save, str(file_name) + '_' + str(self.name) + '.pkl'))
        
        print('Saved:', str(file_name) + '_' + str(self.name) + '.pkl')


        if csv == True:
            self.df_final.to_csv(os.path.join(self.path_to_save, str(file_name) + '_' + str(self.name) + '.csv'))
            
            print('Saved:', str(file_name) + '_' + str(self.name) + '.csv')
