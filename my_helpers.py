import tensorflow as tf
from tensorflow import keras
import numpy as np
import mido


def make_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(units=32, input_shape=(32, )),
        tf.keras.layers.Dense(128, activation=tf.nn.relu),  # regular layer (often Dense is used)
        tf.keras.layers.Dense(128, activation=tf.nn.relu),
        tf.keras.layers.Dense(16, activation=tf.nn.relu)
    ])
    model.compile(optimizer='adam',
                loss='mean_squared_error',
                metrics=['accuracy'])
    
    return model

def generate_more_music(model, start_input, num_bars=120):
    start_input = start_input.reshape(32, 1).T # this is necessary in order to avoid tensorflow rejecting the input
    output = []
    net_out = (np.around(model.predict(start_input))).flatten()
    #print("1111111111 net_out:", net_out)
    previous_input = start_input
    previous_input = previous_input.flatten()
    #print("22222222222 prev_inp", previous_input, np.shape(previous_input))
    for i in range(num_bars-1):
        output.append(list(net_out))
        #print("3333333333 prev_inp 2nd half", previous_input[16:])
        next_input = np.append(previous_input[16:], net_out)
        #print("4444444444 next_inp",next_input, np.shape(next_input))
        next_input = next_input.reshape(32, 1).T
        net_out = (np.around(model.predict(next_input))).flatten()
        previous_input = next_input.flatten()
        #print("555555555 previous_input:", previous_input)

    return output

def write_to_file(bars, ticks_per_beat, filepath="./", filename="reconstructed", stretch=True):
    if stretch:
        bars = np.floor(bars*129)
    ticks_per_sixteenth = round(ticks_per_beat / 4)
    #print(ticks_per_sixteenth, type(ticks_per_sixteenth))
    mid = mido.MidiFile(ticks_per_beat=ticks_per_beat)
    track = mido.MidiTrack()
    mid.tracks.append(track)
    note = -1
    hold = 1
    # make sure all notes are integers (not floats)
    bars = [[int(x) for x in lst] for lst in bars]
    # write integer representation to txt file as well
    f = open(filepath + filename + ".txt", "a")
    f.write("")
    f.close()
    f = open(filepath + filename + ".txt", "a")
    for bar in bars:
        f.write(str(bar)+ "\n")
        #print(bar)
    f.close()
    for bar in bars:
        # print(bar)
        for nt in bar:
            if note is -1:
                note = nt
            elif nt is not 129 and note > 0 and note < 127:
                note_hold_time = hold * ticks_per_sixteenth #* 4
                #print(note_hold_time, type(note_hold_time))
                #print(note, type(note))
                if note > 127 or note < 0:
                    print("what is this madness?", note)
                track.append(mido.Message('note_on', note=note, velocity=100, time=note_hold_time))
                track.append(mido.Message('note_off', note=note, velocity=0, time=0))
                note = nt
                hold = 1
            else:
                if nt < 0 or nt > 129:
                    print("note is outside of range (0-127), something is not right.")
                hold = hold + 1
    mid.save(filepath + filename + ".mid")

def log_evaluation(loss, accurracy, modelnumber):
    f = open("./log.txt", 'a')
    f.write("Params for model " + modelnumber + ": \n")
    f.write("Loss: " + str(loss) + " - Accuracy: " + str(accurracy))

def extract_bars(input_midi, label=None):
    mid = mido.MidiFile(input_midi)
    trks = []
    ticks_per_sixteenth = round(mid.ticks_per_beat / 4)
    for i, track in enumerate(mid.tracks):
        #print('Track {}: {}'.format(i, track.name))
        accu = 0
        note_is_on = False
        bars = []
        current_bar = []
        index_in_bar = 0
        for msg in track:
            # print(msg)
            if msg.type == 'note_on' and not note_is_on and msg.velocity > 0:
                note_is_on = True
                current_bar.append(msg.note)

            if (msg.type == 'note_off' and note_is_on) or(msg.type == 'note_on' and msg.velocity == 0):
                note_is_on = False
                hold_note_for = int(msg.time/ticks_per_sixteenth)
                for i in range(hold_note_for):
                    current_bar.append(129)

            if len(current_bar) >= 16:
                overlap = current_bar[16:]
                while len(current_bar) > 16:
                    current_bar.pop()
                bars.append(current_bar)
                current_bar = overlap

    # assume we just want to use the first track in file
    trks.append(bars)
    return trks[0]
    
def generate_training_data(training_track):
    training_data_input = []
    training_data_target = []
    for i in range(len(training_track)-2):
        training_data_input.append(training_track[i] + training_track[i+1])
        training_data_target.append(training_track[i+2])
    
    training_data_input = np.array(training_data_input)/129
    training_data_target = np.array(training_data_target)/129

    return training_data_input, training_data_target