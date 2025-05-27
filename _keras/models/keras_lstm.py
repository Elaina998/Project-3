import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, LSTM, GRU, Conv1D, Bidirectional, Dropout, RepeatVector, Input, Flatten, BatchNormalization, Activation
from tensorflow.python.keras.layers.convolutional import Conv1D, MaxPooling1D
from tensorflow.keras.layers import Layer, Permute, Lambda, TimeDistributed, add
import tensorflow.keras.backend as K
tf.random.set_seed(2012)

# github https://github.com/JeCase/LoadElectricity_Forecasting_CNN-BiLSTM-Attention/blob/main/Forecasting_CNN-BiLSTM.ipynb


# https://github.com/JeCase/LoadElectricity_Forecasting_CNN-BiLSTM-Attention/blob/main/Forecasting_CNN-BiLSTM%2BAttention.ipynb
def CNN_LSTM_att(structure, args):
    input_shape = (args.seq_len, args.feature_no)
    model = Sequential([
        Conv1D(filters=8, kernel_size=1, activation='relu', input_shape=input_shape),
        BatchNormalization(),
        Conv1D(filters=8, kernel_size=1, activation='relu'),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Bidirectional(LSTM(64, return_sequences=True)),
        Attention(),
        Dropout(structure['dropout']),
        Bidirectional(LSTM(structure['n_hidden_units'])),
        Dense(int(structure['n_hidden_units']/2), activation='relu'),
        Dense(1)
    ])
    return model


def keras_cnn_lstm(structure, args):
    input_shape = (args.seq_len, args.feature_no)
    model = Sequential()
    model.add(Conv1D(filters=8, kernel_size=1, activation='relu', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Dropout(structure['dropout']))
    model.add(Conv1D(filters=8, kernel_size=1, activation='relu'))
    model.add(BatchNormalization())
    model.add(Bidirectional(LSTM(structure["n_hidden_units"], return_sequences=True)))
    model.add(Dropout(structure['dropout']))
    model.add(Bidirectional(LSTM(structure["n_hidden_units"])))
    model.add(Dense(structure["h2"], activation='relu'))
    model.add(Dense(1))
    return model

def encoder_decoder_LSTM(structure, args):
    n_timesteps, n_features, n_outputs = args.seq_len, args.feature_no, args.pred_len
    model = Sequential()
    model.add(LSTM(structure["n_hidden_units"], activation='relu', input_shape=(n_timesteps, n_features)))
    model.add(Dropout(structure["dropout"]))
    model.add(RepeatVector(n_outputs))
    model.add(LSTM(200, activation='relu', return_sequences=True))
    model.add(TimeDistributed(Dense(100, activation='relu')))
    model.add(TimeDistributed(Dense(1)))
    return model


def encoder_decoder_GRU(structure, args):
    n_timesteps, n_features, n_outputs = args.seq_len, args.feature_no, args.pred_len
    model = Sequential()
    model.add(GRU(structure["n_hidden_units"], activation='relu', input_shape=(n_timesteps, n_features)))
    model.add(Dropout(structure["dropout"]))
    model.add(RepeatVector(n_outputs))
    model.add(GRU(structure["h2"], activation='relu', return_sequences=True))
    model.add(TimeDistributed(Dense(int(structure["h2"]/2), activation='relu')))
    model.add(TimeDistributed(Dense(1)))
    return model


class Attention(Layer):
    def __init__(self, return_sequences=True, **kwargs):
        super(Attention, self).__init__()
        self.return_sequences = return_sequences

    def build(self, input_shape):
        self.W = self.add_weight(shape=(input_shape[-1], 1), trainable=True)
        super(Attention, self).build(input_shape)

    def call(self, inputs):
        logits = tf.matmul(inputs, self.W)
        attention_weights = tf.nn.softmax(logits, axis=1)
        weighted_inputs = inputs * attention_weights
        if self.return_sequences:
            return weighted_inputs
        else:
            return tf.reduce_sum(weighted_inputs, axis=1)
            
    def get_config(self):
        config = super(Attention, self).get_config()
        config.update({'return_sequences': self.return_sequences})
        return config


# For model saving/loading compatibility
custom_objects = {
    'Attention': Attention
}


def save_keras_model(model, filepath):
    """Helper function to save Keras model with custom layers"""
    model.save(filepath)
    
    
def load_keras_model(filepath):
    """Helper function to load Keras model with custom layers"""
    return tf.keras.models.load_model(filepath, custom_objects=custom_objects)



