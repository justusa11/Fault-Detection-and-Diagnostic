# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 14:35:17 2024

@author: anoru
"""

##############################################################################
### PACKAGES ###
##############################################################################
import pandas as pd
import numpy as np
import os
import seaborn as sns 
import matplotlib.pyplot as plt
from pandas import Timedelta
import time
from tensorflow import keras
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score

from keras.layers import BatchNormalization
from keras.regularizers import l1_l2
from tensorflow.keras.optimizers.schedules import ExponentialDecay
#%%
##############################################################################
### DEFINE LOCATION ###
##############################################################################

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)

os.chdir(dname)
current_working_folder = os.getcwd()
print(current_working_folder)

#%%
##############################################################################
### LOAD DATA ###
##############################################################################
proven = pd.read_pickle('Proven ML.pkl')

#%%
##############################################################################
### RENAMING COLUMNS ###
##############################################################################

proven.rename(columns={'heat_meter_id': 'Meter id',
                       'volume_m3_demand': 'Volume',
                       'heat_energy_kwh_demand': 'Energy kWh',
                       'supply_temperature': 'Supply',
                       'return_temperature': 'Return',
                       'delta_T': 'dT', 
                       'mean_temp': 'Weather',
                       'fast or slow fault progression Index': 'Ground Truth'}, inplace=True)
print(proven.head())
#%%
##############################################################################
### DROPPING COLUMNS ###
##############################################################################
proven.drop(columns=['Energy kWh', 'Volume', 
                     'Supply', 'dT', 'Weather',
                     'action', 'Fault type', 'Fault type Index',
                     'Is this fast or slow fault progressiong?',
                     'Which season?', 'season index', 'How volume behaves?',
                     'volume behaves index', 'How does energy behaves?', 
                     'How does energy behaves? Index',
                     'Is daily return temperature volatile?',
                     'Is daily return temperature volatile? Index',
                     'Is hourly volume volatile?',
                     'Is hourly volume volatile? Index', 
                     'season Index', 'volume behaves Index'], inplace=True, errors='ignore')

#%%
##############################################################################
### TIME PERIOD and METER ID LIST ###
##############################################################################
proven['failure_start_date'] = pd.to_datetime(proven['failure_start_date'], errors='coerce')
proven['failure_end_date'] = pd.to_datetime(proven['failure_end_date'], errors='coerce')
proven = proven.dropna(subset=['failure_start_date', 'failure_end_date'])
proven['start_window'] = proven['failure_end_date'] - Timedelta(days=90)
proven['end_window'] = proven['failure_end_date']
proven['faulty window'] = proven.apply(lambda row: f"{row['Meter id']}-{row['start_window'].strftime('%Y-%m-%d')}-{row['end_window'].strftime('%Y-%m-%d')}", axis=1)
proven = proven[(proven['Date'] >= proven['start_window']) & (proven['Date'] <= proven['end_window'])]

unique_meter_id = proven['Meter id'].unique().tolist()

#%%
##############################################################################
### PROCESSING ###
##############################################################################
numeric_cols = proven.select_dtypes(include=[np.number]).columns
proven[numeric_cols] = proven[numeric_cols].replace([np.inf, -np.inf], np.nan)
proven[numeric_cols] = proven[numeric_cols].fillna(proven[numeric_cols].mean())
proven = proven.dropna(subset=['Ground Truth'])

label_encoder = LabelEncoder()
label_encoder.fit(proven['Ground Truth'].astype(str))
#%%
##############################################################################
### METRICS ###
##############################################################################
# Start the timer
start_time = time.time()

# Initialize running total of each metric
bas = []  # for balanced accuracy
accs = [] # for accuracy
f1s = [] # for F1 Score
precisions = [] # for Precision
recalls = [] # for Recall
accuracies = []

# Initialize running total of each metric
total_ba = 0
total_acc = 0
total_f1 = 0
total_precision = 0  
total_recall = 0 
total_cm = np.array([[0, 0], [0, 0]]) 
#%%
##############################################################################
### CREATE A TRAINING SEQUENCE ###
##############################################################################
sequence_length = 90
target_column = 'Ground Truth'

feature_columns = proven.columns.drop(['Meter id', target_column])
numeric_feature_columns = proven[feature_columns].select_dtypes(include=[np.number]).columns

scaler = MinMaxScaler()
sequences = []
labels = []

for _, (meter_id, group) in enumerate(proven.groupby('Meter id')):
    group_features = group[numeric_feature_columns]
    if not group_features.empty:
        normalized_features = scaler.fit_transform(group_features)
        for i in range(len(group) - sequence_length + 1):
            sequence = normalized_features[i:i + sequence_length]
            label = group[target_column].iloc[i + sequence_length - 1]
            sequences.append(sequence)
            labels.append(label)

sequences_padded = pad_sequences(sequences, maxlen=sequence_length, padding='post', dtype='float32')
encoded_labels = label_encoder.transform(labels) 

X_train, X_val, y_train, y_val = train_test_split(
    sequences_padded, encoded_labels, test_size=0.1, random_state=42)

#%%
##############################################################################
### MODEL ARCHITECTURE ###
##############################################################################
def create_lstm_model(input_shape, num_classes):
    model = Sequential()
    
    # Adding L1/L2 regularization to LSTM layers
    model.add(LSTM(50, input_shape=input_shape, return_sequences=True, 
                   kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4),
                   recurrent_regularizer=l1_l2(l1=1e-5, l2=1e-4),
                   bias_regularizer=l1_l2(l1=1e-5, l2=1e-4)))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())  # Adding Batch Normalization
    
    model.add(LSTM(50, return_sequences=False,
                   kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4),
                   recurrent_regularizer=l1_l2(l1=1e-5, l2=1e-4),
                   bias_regularizer=l1_l2(l1=1e-5, l2=1e-4)))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())  # Adding Batch Normalization
    
    model.add(Dense(50, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    
    # Setting up an exponential decay learning rate schedule
    lr_schedule = ExponentialDecay(
        initial_learning_rate=0.001,
        decay_steps=10000,
        decay_rate=0.9,
        staircase=True)
    
    model.compile(optimizer=Adam(learning_rate=lr_schedule), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

#%%
##############################################################################
### TRAIN MODEL ###
##############################################################################
input_shape = (X_train.shape[1], X_train.shape[2])
num_classes = len(np.unique(y_train))
model = create_lstm_model(input_shape, num_classes)

early_stopping = EarlyStopping(monitor='val_loss', patience=30, verbose=1, mode='min', restore_best_weights=True)
history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_val, y_val), verbose=1, callbacks=[early_stopping])

#%%
##############################################################################
### METRICS EVALUATION ###
##############################################################################
# Make predictions
y_pred_probs = model.predict(X_val)
y_pred_classes = np.argmax(y_pred_probs, axis=1)

# Calculate metrics
accuracy = accuracy_score(y_val, y_pred_classes)
balanced_accuracy = balanced_accuracy_score(y_val, y_pred_classes)
precision = precision_score(y_val, y_pred_classes, average='weighted')
recall = recall_score(y_val, y_pred_classes, average='weighted')
f1 = f1_score(y_val, y_pred_classes, average='weighted')

# Print metrics
print(f'Accuracy: {accuracy:.4f}')
print(f'Balanced Accuracy: {balanced_accuracy:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1 Score: {f1:.4f}')
#%%
##############################################################################
### SAVE MODEL ###
##############################################################################
model.save('Final1_model.h5')
#%%
##############################################################################
### VISUALIZATION AND ANALYSIS ###
##############################################################################
# Extracting training and validation loss from the history object
train_loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(train_loss) + 1)

# Creating a DataFrame for the loss
loss_df = pd.DataFrame({'Epoch': epochs, 
                        'Training Loss': train_loss, 
                        'Validation Loss': val_loss})

# Plotting training and validation loss
plt.figure(figsize=(10, 6))
sns.lineplot(data=loss_df, x='Epoch', y='Training Loss', label='Training Loss', color='blue')
sns.lineplot(data=loss_df, x='Epoch', y='Validation Loss', label='Validation Loss', color='red')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

#%%
# Extracting training and validation accuracy from the history object
train_accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']

# Creating a DataFrame for the accuracy
accuracy_df = pd.DataFrame({'Epoch': epochs, 
                            'Training Accuracy': train_accuracy, 
                            'Validation Accuracy': val_accuracy})

# Plotting training and validation accuracy
plt.figure(figsize=(10, 6))
sns.lineplot(data=accuracy_df, x='Epoch', y='Training Accuracy', label='Training Accuracy', color='blue')
sns.lineplot(data=accuracy_df, x='Epoch', y='Validation Accuracy', label='Validation Accuracy', color='red')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
#%%
# Assuming metrics are already calculated as shown in the previous steps
metrics_data = {
    'Metric': ['Accuracy', 'Balanced Accuracy', 'Precision', 'Recall', 'F1 Score'],
    'Score': [accuracy, balanced_accuracy, precision, recall, f1]
}
metrics_df = pd.DataFrame(metrics_data)

plt.figure(figsize=(10, 6))
sns.barplot(x='Metric', y='Score', data=metrics_df)
plt.title('Model Performance Metrics')
plt.ylim(0, 1)  # Assuming the scores are between 0 and 1
plt.ylabel('Score')
plt.xlabel('Metric')
plt.show()

#%%
##############################################################################
### LOAD SUSPICION ###
##############################################################################


#%%
# Load suspicion data
#suspicion = pd.read_pickle('Suspicion.pkl')
#suspicion_path = r""
suspicion = pd.read_pickle(suspicion_path)
#%%
##############################################################################
### RENAMING COLUMNS ###
##############################################################################

suspicion.rename(columns={'heat_meter_id': 'Meter id',
                       'volume_m3_demand': 'Volume',
                       'heat_energy_kwh_demand': 'Energy kWh',
                       'supply_temperature': 'Supply',
                       'return_temperature': 'Return',
                       'delta_T': 'dT', 
                       'mean_temp': 'Weather'}, inplace=True)
print(suspicion.head())
#%%
##############################################################################
### TIME PERIOD and METER ID LIST ###
##############################################################################

suspicion['failure_start_date'] = pd.to_datetime(suspicion['failure_start_date'], errors='coerce')
suspicion['failure_end_date'] = pd.to_datetime(suspicion['failure_end_date'], errors='coerce')
suspicion = suspicion.dropna(subset=['failure_start_date', 'failure_end_date'])
suspicion['start_window'] = suspicion['failure_end_date'] - Timedelta(days=90)
suspicion['end_window'] = suspicion['failure_end_date']
suspicion['faulty window'] = suspicion.apply(lambda row: f"{row['Meter id']}-{row['start_window'].strftime('%Y-%m-%d')}-{row['end_window'].strftime('%Y-%m-%d')}", axis=1)
suspicion = suspicion[(suspicion['Date'] >= suspicion['start_window']) & (suspicion['Date'] <= suspicion['end_window'])]

unique_meter_id = suspicion['Meter id'].unique().tolist()
#%%
# Dropping rows with missing values
suspicion.dropna(subset=['Return'], inplace=True)

# Normalizing the 'Return' temperature
suspicion['Return'] = scaler.transform(suspicion[['Return']])

# Print
print(suspicion.head())
#%%
##############################################################################
### CREATE SEQUENCES FOR SUSPICION DATA ###
##############################################################################
suspicion_sequences = []

# Assuming 'Meter id' and 'Date' columns are available for grouping and sorting
for meter_id, group in suspicion.groupby('Meter id'):
    group = group.sort_values(by='Date')  # Sorting by date
    for i in range(len(group) - sequence_length + 1):
        sequence = group['Return'].iloc[i:i + sequence_length].values
        suspicion_sequences.append(sequence.reshape(-1, 1))

suspicion_sequences_padded = pad_sequences(suspicion_sequences, maxlen=sequence_length, padding='post', dtype='float32')
#%%
##############################################################################
### PREDICT ON SUSPICION DATA ###
##############################################################################
# Load the trained model (if not already in memory)
model = keras.models.load_model('Final1_model.h5')

# Making predictions
suspicion_pred_probs = model.predict(suspicion_sequences_padded)
suspicion_pred_classes = np.argmax(suspicion_pred_probs, axis=1)

# Decode predictions if needed
suspicion_predictions = label_encoder.inverse_transform(suspicion_pred_classes)