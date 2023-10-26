"""
Generate 2D feature matrix using SMILES 
Author: Son Gyo Jung
Email: sgj13@cam.ac.uk
"""

import numpy as np
import pandas as pd
import re
import joblib
import pickle
import os

from rdkit import Chem
from rdkit.Chem import rdchem
from rdkit.Chem import AllChem

from deepchem.feat.smiles_tokenizer import BasicSmilesTokenizer


class load():
    '''
    Class used to generate features using several descriptors which take SMILES as input
    args: 
        (1) path_to_file (type:str) - location of the SMILES data file in CSV format
        (2*)
    return: 
        (1) DataFrame of chemical features
    '''

    def __init__(self, path_to_file: str, *args, **kwargs):

        smile_col = kwargs.get('smile_col')
        
        target_col = kwargs.get('target_col')
        
        smile_in_index = kwargs.get('target_col')


        try:
            
            self.df = pd.read_csv(path_to_file)
            
        except:
            
            self.df = joblib.load(path_to_file)


        if all(v is not None for v in [smile_col, target_col]):
            
            # Column names
            self.smiles = self.df.columns.values[smile_col]
            
            self.target = self.df.columns.values[target_col]


        elif all(v is not None for v in [smile_in_index, target_col]):
            
            # Column names
            self.df.insert(0, 'smiles', self.df.index)
            
            self.smiles = self.df.columns.values[0]
            
            self.target = self.df.columns.values[target_col]
            

        else:
            
            # Column names
            self.smiles = self.df.columns.values[0]
            
            self.target = self.df.columns.values[1]


        print('Name of SMILES column:', self.smiles)
        print('Name of target column: ', self.target)
        print('No. of SMILES: ', len(self.df))


        # List of features for reference
        chem = ['H', 'C', 'O', 'N', 'P', 'S', 'other_elements']
        
        chem_prop = ['NumHs', 'Degree', 'Charge', 'Valence', 'Ring', 'Aromatic']
        
        chrial_prop = ['CHI_TETRAHEDRAL_CW', 'CHI_TETRAHEDRAL_CCW', 'CHI_OTHER']
        
        hybrid_prop = ["S", "SP", "SP2", "SP3", "SP3D", "SP3D2", "OTHER"]
        
        symbol = ['(', ')', '[', ']', '.', ':', '=', '#', '\\', '/', '@', '+', '-']
        
        rest = ['Ion_charge_2','Ion_charge_3','Ion_charge_4','Ion_charge_5','Ion_charge_6','Ion_charge_7', 'Ring_start', 'Ring_end']
        

        self.full_features = chem + chem_prop + chrial_prop + hybrid_prop + symbol + rest


        name_list = ['H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', \
                    'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', \
                    'Kr', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', \
                    'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', \
                    'Yb', 'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', \
                    'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr', \
                    'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg', 'Cn', 'Fl', 'Mc', 'Lv', 'Ts', 'Og']

                    #'Nh' removed due to confusion with nH


        self.chem_list = []


        for i in name_list:
            
            if len(i) == 2:
                
                self.chem_list.append(i.lower())
        


    def movecol(self, df: pd.DataFrame, cols_to_move: list, ref_col: str, place: str = 'After') -> pd.DataFrame:
    
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

    
    
    def diagram(self, smiles: str):
        '''
        Generate diagram of the chemical structure
        '''
        
        mol = Chem.MolFromSmiles(smiles)
        
        print('Chemical compound: ', smiles)


        return mol



    def smi_atom_matrix(self, atom) -> list:
        '''
        Generate atom-based features
        '''
        
        Chiral = {"CHI_UNSPECIFIED":0,  "CHI_TETRAHEDRAL_CW":1, "CHI_TETRAHEDRAL_CCW":2, "CHI_OTHER":3}
        
        Hybridization = {"UNSPECIFIED":0, "S":1, "SP":2, "SP2":3, "SP3":4, "SP3D":5, "SP3D2":6, "OTHER":7}
        
        chem = ['H', 'C', 'O', 'N', 'P', 'S']
        
        atom_features = 7*[0]
        
        
        try:
            # Get chemical symbol and corresponding index
            j = chem.index(atom.GetSymbol())

            atom_features[j] = 1
        
        except:
            # If it does not exist in chem, select others
            atom_features[6] = 1


        # Chirality
        # Initial array initialized with zeros    
        chirality =  [0]*(len(Chiral)-1)
        
        if Chiral.get(str(atom.GetChiralTag()), 0) != 0:
            
            chirality[Chiral.get(str(atom.GetChiralTag()), 0)] = 1
        
        
        # Hybridization
        hybridization =  [0]*(len(Hybridization)-1)

        if Hybridization.get(str(atom.GetHybridization()), 0) != 0:
            
            hybridization[Hybridization.get(str(atom.GetHybridization()), 0)] = 1
        
        
        # Normalised features
        atom_features.extend([
                                #atom.GetSymbol(), 
                                atom.GetTotalNumHs(),
                                atom.GetTotalDegree()/4,
                                atom.GetFormalCharge()/8,
                                atom.GetTotalValence()/8,
                                atom.IsInRing()*1,
                                atom.GetIsAromatic()*1] 
                                + chirality 
                                + hybridization
                            )
        
        #print(atom_features, len(atom_features))
        
        return atom_features 



    def smi_stucture_matrix(self, smi: str, check: int, ref: list):
        '''
        Generate structure-based features
        '''

        # Symbols to consider
        symbol = ['(', ')', '[', ']', '.', ':', '=', '#', '\\', '/', '@', '+', '-']

        struc_feature = 21*[0]
        

        try:
            
            # Identify symbol and corresponding index
            j = symbol.index(smi)
            
            struc_feature[j] = 1


            # if the symbol is either + or -, need to check for subsequent symbol/number
            if j == 11 or j == 12:
                
                check = 1
                
            else:
                
                check = 0


        except:
            
            if smi.isdigit() == True:
                
                if check == 0 and smi not in ref:
                    
                    # Ring start
                    struc_feature[19] = 1
                    
                    ref.append(smi)


                elif check == 0 and smi in ref:
                    
                    # Ring end
                    struc_feature[20] = 1


                elif check == 1:
                    # If the previous symbol was + or -, then the number following it is the ionic charge (2-7)
                    z = int(smi) - 1 + 12
                    
                    struc_feature[z] = 1

                    check = 0

                else:
                    
                    pass
        
        
        return struc_feature, check, ref



    def generate_feature_matrix(self, smiles: str) -> list:
        '''
        Generate full features with specified logics
        '''
        
        no_atom_features = 23
        
        no_struc_features = 21
        
        ref, feature, token_list = [], [], []
        
        i, check = 0, 0
        
        matrix = []
        
        tokenizer = BasicSmilesTokenizer()
        
        smiles_token = tokenizer.tokenize(smiles)

        mol = Chem.MolFromSmiles(smiles)
        
        
        for token in smiles_token:
            
            if len(token) == 1:
                
                token_list.append(token)
                
                
            elif len(token) > 1 and token[0] == '[':
                
                if any(map(str(token.lower()).__contains__, self.chem_list)) == False:
                    
                    for char in token:
                        
                        token_list.append(char)
                        
                        
                else:
                    
                    for l in range(len(token)-1):
                        
                        substring = token[l:l+2]
                        
                        
                        if substring.isalpha() == True:
                            
                            if substring.lower() in self.chem_list:
                                
                                for i in range(l):
                                    
                                    token_list.append(token[i])


                                token_list.append(substring)

                                token = token[l:].replace(substring,'')

                                for char in token:
                                    
                                    token_list.append(char)
                                    
                                    continue

            else:
                
                # if len(token) == 2
                token_list.append(token)
                    
                    
        print(smiles)
        print('Number of tokens:', len(token_list))


        for smi in token_list:
            
            if smi.isalpha() == True: 
                
                #print(smi)
                if smi == 'H':
                    
                    # Hydrogen
                    feature.extend([1,0,0,0,0,0,0])

                    # Padding
                    feature.extend(16*[0])
                    
                else:
                    
                    # Generate atom-based features
                    feature.extend(self.smi_atom_matrix(rdchem.Mol.GetAtomWithIdx(mol, i)))
                    
                    i = i + 1
                    
                    
                # Padding
                feature.extend(no_struc_features*[0])
                
                matrix.append(feature)
                
                feature = []
                
                
            else:
                # otherwise must be structure-based features
                # Padding
                feature.extend(no_atom_features*[0])
                
                struc_feature, check, ref = self.smi_stucture_matrix(smi, check, ref)
                
                feature.extend(struc_feature)
                
                matrix.append(feature)
                
                feature = []
                
                
        print('Number of features:', len(matrix))


        self.mismatch = 0

        if len(matrix) != len(token_list):
            
            self.mismatch += 1
            
            print('Mismatch identified for: ', smiles)
            

        return matrix, token_list



    def generate(self, save: bool, *args, **kwargs) -> dict:
        '''
        Run the featurization
        '''
        
        file_name = kwargs.get('file_name')
        
        n = kwargs.get('n')

        dic = {}
        
        dic_problem = {}

        i = 0
        

        if n != None:
            
            smile_list = self.df[self.smiles].tolist()[:n]
            
        elif n == None:
            
            smile_list = self.df[self.smiles].tolist()


        for smi in smile_list:
            
            try:
                
                matrix, token_list = self.generate_feature_matrix(smi)

                
                df_temp = pd.DataFrame(
                                        data=matrix,
                                        index=token_list,
                                        columns=self.full_features
                                        ) 

                dic[i] = df_temp

                i = i + 1
                
                print()
                
                
            except:
                
                dic_problem[i] = smi
                
                i = i + 1


        if save == True:
            
            if file_name is not None:
                
                with open('smiles_feature_matrix_' + str(file_name) + '.pkl', 'wb') as f:
                    
                    pickle.dump(dic, f)

                    print('Saved file named: smiles_feature_matrix_' + str(file_name) + '.pkl')
                    
                    
                with open('smiles_feature_matrix_error_' + str(file_name) + '.pkl', 'wb') as f:
                    
                    pickle.dump(dic_problem, f)

                    print('Saved file named: smiles_feature_matrix_error_' + str(file_name) + '.pkl')


            else:
                with open('smiles_feature_matrix.pkl', 'wb') as f:
                    
                    pickle.dump(dic, f)

                    print('Saved file named: smiles_feature_matrix.pkl')
                    
                with open('smiles_feature_matrix_error.pkl', 'wb') as f:
                    
                    pickle.dump(dic_problem, f)

                    print('Saved file named: smiles_feature_matrix_error.pkl')


        print('No of SMILES processed: ', len(dic))
        print('Total no. of mismatch: ', self.mismatch)


        return dic, self.df
