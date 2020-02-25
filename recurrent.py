import tensorflow as tf
#import mitdeeplearning as mdl 
import numpy as np
import os
import time
import functools
from IPython import display as ipythondisplay
from tqdm import tqdm 
import my_helpers as mh

#assert len(tf.config.list_physical_devices('GPU')) > 0

song1 = mh.extract_bars('./inputs/inp_1_cMaj_4-4th_temp_1.mid')
song2 = mh.extract_bars('./inputs/inp_2_cMaj_4-4th_temp_1.mid')
song3 = mh.extract_bars('./inputs/inp_3_cMaj_4-4th_temp_1.mid')
song4 = mh.extract_bars('./inputs/inp_4_cMaj_4-4th_temp_1.mid')
song5 = mh.extract_bars('./inputs/inp_5_cMaj_4-4th_temp_1.mid')

songs = np.array(song1 + song2 + song3 + song4 + song5).flatten()

def get_batch(song_list, seq_length, batch_size):
    # get length of inputs and randomly take subset
    n = song_list.shape[0] - 1
    idx = np.random.choice(n-seq_length, batch_size)
    # create matching subsets of input and output strings for rnn
    input_batch =  [song_list[i : i+seq_length] for i in idx]
    output_batch = [song_list[i+1 : i+seq_length+1] for i in idx]
    #create batches in proper size to feed to rnn
    x_batch = np.reshape(input_batch, [batch_size, seq_length])
    y_batch = np.reshape(output_batch, [batch_size, seq_length])

    return x_batch, y_batch

x_batch, y_batch = get_batch(songs, 5, 1)

# standard lstm stolen from the internet
def LSTM(rnn_units): 
    return tf.keras.layers.LSTM(
        rnn_units, 
        return_sequences=True, 
        recurrent_initializer='glorot_uniform',
        recurrent_activation='sigmoid',
        stateful=True,)

def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
    # vocab size is the number of possible values the inputs (and outputs) can take on
    # i. e. the number of unique characters (or numbers) in the data set 
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim, batch_input_shape=[batch_size, None]),
        LSTM(rnn_units),
        tf.keras.layers.Dense(vocab_size)
    ])
    return model

#model = build_model(vocab_size, embedding_dim=256, rnn_units=1024, batch_size=32)

# custom loss functions

def compute_loss(labels, logits):
    loss = tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)
    return loss

# training parameters

num_training_iterations = 2000
batch_size = 4
seq_length = 100
learning_rate = 5e-3

vocab_size = 130 # number of unique characters in dataset
embedding_dim = 256
rnn_units = 1024

checkpoint_dir = './rnn_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "checkpoint")

model = build_model(vocab_size, embedding_dim, rnn_units, batch_size)
optimizer = tf.keras.optimizers.Adam(learning_rate)

# custom training step 
def train_step(x, y): 
    with tf.GradientTape() as tape:
        y_hat = model(x) 
        loss = compute_loss(y, y_hat) 

    grads = tape.gradient(loss, model.trainable_variables) 
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss

history = []

# run the network for a while
for iter in tqdm(range(num_training_iterations)):
    x_batch, y_batch = get_batch(songs, seq_length, batch_size)
    print(x_batch.shape, y_batch.shape)
    loss = train_step(x_batch, y_batch)

    history.append(loss.numpy().mean())

    if iter % 100 == 0:     
      model.save_weights(checkpoint_prefix)

def generate_song(model, start_string, generation_length=1000):
    song_generated = []
    current_string = start_string
    # Here batch size == 1
    model.reset_states()
    tqdm._instances.clear()
    for i in tqdm(range(generation_length)):
      predictions = model(current_string)
      predictions = tf.squeeze(predictions, 0)
      #dont think I need this
      #predicted_id = tf.random.categorical(predictions, num_samples=1)[-1,0].numpy()
      #song_generated = tf.expand_dims([predicted_id], 0)
      current_string = predictions
      song_generated.append(predictions)

    return (start_string + ''.join(song_generated))