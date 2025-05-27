import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, LSTM, Conv1D, Dropout, Bidirectional, Multiply, Concatenate
from tensorflow.keras.layers import Permute, Lambda, RepeatVector, Flatten
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K

SINGLE_ATTENTION_VECTOR = False
def attention_3d_block(inputs):
    # inputs.shape = (batch_size, time_steps, input_dim)
    input_dim = int(inputs.shape[2])
    a = inputs
    #a = Permute((2, 1))(inputs)
    #a = Reshape((input_dim, TIME_STEPS))(a) # this line is not useful. It's just to know which dimension is what.
    a = Dense(input_dim, activation='softmax')(a)
    if SINGLE_ATTENTION_VECTOR:
        a = Lambda(lambda x: K.mean(x, axis=1), name='dim_reduction')(a)
        a = RepeatVector(input_dim)(a)
    a_probs = Permute((1, 2), name='attention_vec')(a)

    #output_attention_mul = merge([inputs, a_probs], name='attention_mul', mode='mul')
    output_attention_mul = Multiply()([inputs, a_probs])
    return output_attention_mul

# Another way of writing the attention mechanism is suitable for the use of the above error source:https://blog.csdn.net/uhauha2929/article/details/80733255
def attention_3d_block2(inputs, single_attention_vector=False):
    # If the upper layer is LSTM, you need return_sequences=True
    # inputs.shape = (batch_size, time_steps, input_dim)
    time_steps = K.int_shape(inputs)[1]
    input_dim = K.int_shape(inputs)[2]
    a = Permute((2, 1))(inputs)
    a = Dense(time_steps, activation='softmax')(a)
    if single_attention_vector:
        a = Lambda(lambda x: K.mean(x, axis=1))(a)
        a = RepeatVector(input_dim)(a)

    a_probs = Permute((2, 1))(a)
    # Multiplied by the attention weight, but there is no summation, it seems to have little effect
    # If you classify tasks, you can do Flatten expansion
    # element-wise
    output_attention_mul = Multiply()([inputs, a_probs])
    return output_attention_mul

def improved_attention_mechanism(inputs, num_heads=4):
    input_dim = int(inputs.shape[2])
    head_dim = input_dim // num_heads
    
    # Multi-head attention
    attention_heads = []
    for i in range(num_heads):
        head = Dense(head_dim)(inputs)
        query = Dense(head_dim)(head)
        key = Dense(head_dim)(head)
        value = Dense(head_dim)(head)
        
        # Scaled dot-product attention
        attention_scores = tf.matmul(query, tf.transpose(key, [0, 2, 1]))
        attention_scores = attention_scores / tf.math.sqrt(tf.cast(head_dim, tf.float32))
        attention_weights = tf.nn.softmax(attention_scores, axis=-1)
        attention_output = tf.matmul(attention_weights, value)
        attention_heads.append(attention_output)
    
    # Concatenate heads and project
    multi_head = Concatenate(axis=-1)(attention_heads)
    output = Dense(input_dim)(multi_head)
    return output

def CNN_BiLSTM_attention(structure, args):
    """
    CNN-BiLSTM with attention mechanism
    
    Args:
        structure: Dictionary containing model hyperparameters
        args: Arguments containing sequence length and feature count
    """
    # Get input shape from args
    input_shape = (args.seq_len, args.feature_no)
    
    # Create model using Functional API
    inputs = Input(shape=input_shape)
    
    # CNN layer
    x = Conv1D(
        filters=8,  # Using standard 8 filters like other models
        kernel_size=1,
        activation='relu'
    )(inputs)
    
    x = Dropout(structure['dropout'])(x)
    
    # BiLSTM layer
    lstm_units = structure['n_hidden_units']
    lstm_out = Bidirectional(LSTM(lstm_units, return_sequences=True))(x)
    lstm_out = Dropout(structure['dropout'])(lstm_out)
    
    # Apply attention
    attention_mul = attention_3d_block(lstm_out)
    attention_mul = Flatten()(attention_mul)
    
    # Output layers
    x = Dense(structure['h2'], activation='relu')(attention_mul)
    output = Dense(1)(x)
    
    # Create and return model
    model = Model(inputs=inputs, outputs=output)
    return model