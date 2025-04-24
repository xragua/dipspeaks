#PKGS
##########################################################################################
##########################################################################################
##########################################################################################
# Import Libraries
# Standard libraries

import warnings

# Data manipulation and analysis
import numpy as np
import pandas as pd

# Plotting and visualization
import matplotlib.pyplot as plt

# Scikit-learn for machine learning
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, Normalizer

# TensorFlow and Keras for deep learning
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model

# Ignore warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', message='divide by zero encountered in divide')

#########################################################################################
pipeline = Pipeline([
    ('normalizer', Normalizer()),
    ('scaler', MinMaxScaler())
])

from .helper_functions import (
     rebin_snr,
     _moving_average,
    _base_calculator,
)

from .detection import (
    _detection,
)

from .synthetic_data_for_train import (
    _calculate_synthetic_data,
)


from .evaluation import(
    _modified_z_score,
    _outlier_probability,
    _real_probability,
)



def _clean_autoencoder(pd_to_clean, pd_base, show_plot_eval, show_plot=True):
    '''
    Clean a dataset by detecting and flagging outliers using an autoencoder model.

    This function builds an autoencoder to reconstruct data from a baseline dataset (`pd_base`).
    It then uses the reconstruction errors to identify outliers in a separate dataset (`pd_to_clean`)
    based on the specified features. 

    Parameters:
    - pd_to_clean (pd.DataFrame): The dataset to be cleaned, containing columns that match those in
                                    `pd_base` for feature selection and comparison.
    - pd_base (pd.DataFrame): The baseline dataset used to train the autoencoder, typically representing
                                normal or expected data patterns.

    Returns:
    - pd_to_clean (pd.DataFrame): The input dataframe with additional columns:
        - **zscores** (`float`): z-score of each sample’s reconstruction error  
        - **error_percentile** (`float`): percentile rank of the reconstruction errors  
    - mse_test (np.array): Mean squared reconstruction errors for the `pd_to_clean` dataset.
    - mse_train (np.array): Mean squared reconstruction errors for the `pd_base` dataset.
    '''
    # Select relevant columns
    selected_columns = ['prominence', 'duration','density','snr']
    pd_to_clean_selection = pd_to_clean[selected_columns]
    pd_base_selection = pd_base[selected_columns]

    # Split data into training (baseline) and test (to_clean) sets
    X_train = pd_base_selection
    X_eval = pd_to_clean_selection

    # Fit the pipeline on X_train and transform both X_train and X_eval
    pipeline.fit(X_train)
    X_train_transformed =X_train# pipeline.transform(X_train)
    X_eval_transformed = X_eval#pipeline.transform(X_eval)

    # Define the autoencoder architecture
    input_dim = X_train_transformed.shape[1]
    encoding_dim =256

    input_layer = Input(shape=(input_dim,))
    x = Dense(encoding_dim,          activation='elu')( input_layer)
    x = Dense(encoding_dim // 2,     activation='elu')(x)
    x = Dense(encoding_dim // 4,     activation='elu')(x)
    x = Dense(encoding_dim // 8,     activation='elu')(x)
    latent = Dense(encoding_dim // 16, activation='elu', name='latent')(x)

    # ----- Decoder (simétrico → dimensiones crecen) -----
    x = Dense(encoding_dim // 8,  activation='elu')(latent)
    x = Dense(encoding_dim // 4,  activation='elu')(x)
    x = Dense(encoding_dim // 2,  activation='elu')(x)
    decoded = Dense(input_dim,     activation='elu')(x)  # o 'sigmoid'/'linear'

    autoencoder = Model(input_layer, decoded)

    # Compile the autoencoder
    autoencoder.compile(optimizer='adam', loss='mae')

    # Define callbacks for early stopping and learning rate adjustment
    early_stopping = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=5, min_lr=0.00001)

    # Train the autoencoder on the baseline data
    history = autoencoder.fit(X_train_transformed, X_train_transformed,
                                epochs=20000,
                                batch_size=128,
                                shuffle=True,
                                validation_split=0.1,
                                callbacks=[early_stopping, reduce_lr],
                                verbose=0)
    if show_plot:
        # Plot the learning curve
        plt.figure(figsize=(4, 3))
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Learning Curve')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

    # Reconstruct data and compute reconstruction errors
    reconstructed_train = autoencoder.predict(X_train_transformed)
    reconstructed_test = autoencoder.predict(X_eval_transformed)

    # Calculate Mean Squared Error (MSE) for reconstruction errors
    mse_train = np.mean(np.power(X_train_transformed - reconstructed_train, 2), axis=1)
    mse_test = np.mean(np.power(X_eval_transformed - reconstructed_test, 2), axis=1)

    # Identify outliers using reconstruction error
    zscores, error_percentile = _outlier_probability(mse_train, mse_test, np.array(pd_to_clean.t), show_plot=show_plot, show_plot_eval=False)

    # Append outlier probability and flags to the cleaned dataset
    pd_to_clean['zscores'] = zscores
    pd_to_clean['error_percentile'] = error_percentile
    pd_to_clean = pd_to_clean.sort_values(by='t').reset_index(drop=True)

    return pd_to_clean, mse_test, mse_train 