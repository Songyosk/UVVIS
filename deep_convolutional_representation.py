'''
Create CNN based on ResNet-50 to generate flatten representation of the 2D feature matrix
'''
import numpy as np
import pandas as pd
from pyparsing import And
import os

from tensorflow.keras import activations, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, Add, MaxPooling2D, ZeroPadding2D, AveragePooling2D
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint, Callback


class prepare():
    '''
    Class used to create CNN which take SMILES feature matrice as inputs
    
    args: 
        (1) path_to_save (type:str) - location to save new data files
    '''

    def __init__(self, path_to_save: str, *args, **kwargs):
        self.path_to_save = path_to_save
        
        os.chdir(self.path_to_save)


    def identity_block(self, x, filters): 
        '''
        Identity block
        '''
        
        x_skip = x 
        
        f1, f2 = filters


        # First block 
        x = Conv2D(f1, kernel_size=(1, 1), strides=(1, 1), padding='valid', kernel_regularizer=l2(0.001))(x)
        
        x = BatchNormalization()(x)
        
        x = Activation(activations.relu)(x)


        # Second block 
        x = Conv2D(f1, kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_regularizer=l2(0.001))(x)
        
        x = BatchNormalization()(x)
        
        x = Activation(activations.relu)(x)
        

        # Third block 
        x = Conv2D(f2, kernel_size=(1, 1), strides=(1, 1), padding='valid', kernel_regularizer=l2(0.001))(x)
        
        x = BatchNormalization()(x)


        # Skip connection addition
        x = Add()([x, x_skip])
        
        x = Activation(activations.relu)(x)


        return x

    

    def convolutional_block(self, x, s, filters):
        '''
        Convolutional block
        ''' 
        
        x_skip = x
        
        f1, f2 = filters


        # First block
        x = Conv2D(f1, kernel_size=(1, 1), strides=(s, s), padding='valid', kernel_regularizer=l2(0.001))(x)
        
        x = BatchNormalization()(x)
        
        x = Activation(activations.relu)(x)
        

        # Second block
        x = Conv2D(f1, kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_regularizer=l2(0.001))(x)
        
        x = BatchNormalization()(x)
        
        x = Activation(activations.relu)(x)
        

        # Third block
        x = Conv2D(f2, kernel_size=(1, 1), strides=(1, 1), padding='valid', kernel_regularizer=l2(0.001))(x)
        
        x = BatchNormalization()(x)


        # Skip connection   
        x_skip = Conv2D(f2, kernel_size=(1, 1), strides=(s, s), padding='valid', kernel_regularizer=l2(0.001))(x_skip)
        
        x_skip = BatchNormalization()(x_skip)
        

        # Skip connection addition 
        x = Add()([x, x_skip])
        
        x = Activation(activations.relu)(x)
        

        return x



    def create_model(self, x_train, y_train, *args, **kwargs):
        '''
        Create deep model
        '''
        
        init_kernel_size = kwargs.get('init_kernel_size') 
        
        input_tensor = Input(shape=(x_train.shape[1], x_train.shape[2], 1)) 

        x = ZeroPadding2D(padding=(3, 3))(input_tensor)


        # Input stage
        if init_kernel_size is None:
            
            x = Conv2D(64, kernel_size=(3, 3), strides=(2, 2))(x)


        elif init_kernel_size is not None:
            
            x = Conv2D(64, kernel_size=init_kernel_size, strides=(2, 2))(x)


        x = BatchNormalization()(x)
        
        x = Activation(activations.relu)(x)
        
        x = MaxPooling2D((3, 3), strides=(2, 2))(x)
        

        # First block
        x = self.convolutional_block(x, s=1, filters=(64, 256))


        # Second block
        x = self.convolutional_block(x, s=2, filters=(128, 512))
        
        x = self.identity_block(x, filters=(128, 512))
        
        x = self.identity_block(x, filters=(128, 512))


        # Third block
        x = self.convolutional_block(x, s=2, filters=(256, 1024))
        
        x = self.identity_block(x, filters=(256, 1024))
        
        x = self.identity_block(x, filters=(256, 1024))
        
        x = self.identity_block(x, filters=(256, 1024))
        

        # Fourth block
        x = self.convolutional_block(x, s=2, filters=(512, 2048))


        # Average pooling & dense connection
        x = AveragePooling2D((2, 2), padding='same')(x)
        
        x = Flatten()(x)
        
        x = Dense(512, activation='relu', name='deep_representation')(x)

        output_tensor = Dense(1, activation='linear', name='output')(x)  


        # Define the model 
        model = Model(inputs=input_tensor, outputs=output_tensor, name='Deep_CNN')

        model.summary()

        lr_schedule = ExponentialDecay(
                            initial_learning_rate=1e-2,
                            decay_steps=10000,
                            decay_rate=0.9
                            )

        model.compile(
                optimizer=Adam(lr_schedule), 
                #optimizer=Adam(1e-2), 
                loss='mse', 
                metrics=['mse', 'mae']
                )
        

        return model



    def create_simpler_model(self, x_train, y_train , *args, **kwargs):
        '''
        Create model
        '''
        
        init_kernel_size = kwargs.get('init_kernel_size') # saved weight

        input_tensor = Input(shape=(x_train.shape[1], x_train.shape[2], 1)) 

        x = ZeroPadding2D(padding=(3, 3))(input_tensor)

        # 1st stage
        # here we perform maxpooling
        if init_kernel_size is None:
            x = Conv2D(64, kernel_size=(3, 3), strides=(2, 2))(x)

        elif init_kernel_size is not None:
            x = Conv2D(64, kernel_size=init_kernel_size, strides=(2, 2))(x)
        x = BatchNormalization()(x)
        x = Activation(activations.relu)(x)
        x = MaxPooling2D((3, 3), strides=(2, 2))(x)

        #2nd stage 
        # frm here on only conv block and identity block, no pooling
        x = self.convolutional_block(x, s=1, filters=(64, 256))

        # 3rd stage
        x = self.convolutional_block(x, s=2, filters=(128, 512))

        # 4th stage
        x = self.convolutional_block(x, s=2, filters=(256, 1024))

        # 5th stage
        x = self.convolutional_block(x, s=2, filters=(512, 2048))

        # ends with average pooling and dense connection
        x = AveragePooling2D((2, 2), padding='same')(x)
        x = Flatten()(x)
        x = Dense(512, activation='relu')(x)

        output_tensor = Dense(1, activation='linear')(x)  

        # define the model 
        model = Model(inputs=input_tensor, outputs=output_tensor, name='CNN')

        #model.summary()

        lr_schedule = ExponentialDecay(
                            initial_learning_rate=1e-2,
                            decay_steps=10000,
                            decay_rate=0.9
                            )

        model.compile(
                optimizer=Adam(lr_schedule), 
                #optimizer=Adam(1e-2), 
                loss='mse', 
                metrics=['mse', 'mae']
                )

        return model



    def resnet(
                self, 
                x_train: np.array, 
                y_train: np.array, 
                x_val: np.array, 
                y_val: np.array, 
                maximum_epochs: int, 
                early_stop_epochs: int, 
                deeper_model: bool, 
                *args, 
                **kwargs
                ):
        '''
        Create ResNet-inspired CNN Model
        '''
        
        init_kernel_size = kwargs.get('init_kernel_size') 
        
        checkpoint_path = kwargs.get('checkpoint_path') 
        
        save_best_only = kwargs.get('save_best_only') 


        if deeper_model == True:
            
            model = self.create_model(x_train, y_train, init_kernel_size)
            
        else:
            
            model = self.create_simpler_model(x_train, y_train, init_kernel_size)


        # To use trained weights
        if checkpoint_path != None:
            
            model.load_weights(checkpoint_path)


        if x_val is not None and y_val is not None:
            
            # Naming convention to use for saved weights
            checkpoint_name = 'Weights-{epoch:03d}--{loss:.2f}--{val_loss:.2f}.hdf5' 


            # Define callbacks 
            callbacks_list = [
                                EarlyStopping(
                                            monitor='val_loss', 
                                            patience=early_stop_epochs
                                            ),  

                                ModelCheckpoint(
                                                filepath=checkpoint_name,
                                                monitor='val_loss', 
                                                save_best_only=save_best_only
                                                )
                            ]    


            # Fit the model
            blackbox = model.fit(
                                x=x_train, 
                                y=y_train,
                                batch_size=32,
                                epochs=maximum_epochs,
                                validation_data=(x_val, y_val),
                                shuffle=True, 
                                verbose=1,
                                callbacks=callbacks_list
                                )


        # Otherwise training the model with full training dataset
        elif x_val is None and y_val is None:
            
            # Naming convention to use for saved weights
            checkpoint_name = 'Weights-{epoch:03d}--{loss:.2f}.hdf5' 


            # Define callbacks
            callbacks_list = [
                                EarlyStopping(
                                            monitor='loss', 
                                            patience=early_stop_epochs
                                            ),  

                                ModelCheckpoint(
                                                filepath=checkpoint_name,
                                                monitor='loss', 
                                                save_best_only=save_best_only
                                                )
                            ]    


            # Fit the model
            blackbox = model.fit(
                                x=x_train, 
                                y=y_train,
                                batch_size=32,
                                epochs=maximum_epochs,
                                shuffle=True, 
                                verbose=1,
                                callbacks=callbacks_list
                                )
            

        return model