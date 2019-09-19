import mido
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt


print(tf.__version__)


mid = mido.MidiFile('MIDI_sample.mid')

# midi tempo is microseconds per beat. standard is 500000
# message time attribute is number of ticks to wait
print(mid.ticks_per_beat) # ticks are midi time unit. if tempo is 500000 and ticks per beat is 480 then there are ~1042
# microseconds per tick
ticks_per_sixteenth = round(mid.ticks_per_beat / 4)
trks = []

# read in file and transform to nn representation
for i, track in enumerate(mid.tracks):
    print('Track {}: {}'.format(i, track.name))
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

    for bar in bars:
        print(bar, len(bar))



    trks.append(bars)

# generate training data
training_data_input = []
training_data_target = []
# just take first track here for convenience sake
training_track = trks[4] # hardcoded for now, later take first non empty track
for i in range(len(training_track)-2):
    #print(training_track[i], training_track[i+1], training_track[i+2])
    training_data_input.append(training_track[i] + training_track[i+1])
    training_data_target.append(training_track[i+2])

training_data_input = np.array(training_data_input)
training_data_target = np.array(training_data_target)

# train NN
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(units=32, input_shape=(32, )),
    tf.keras.layers.Dense(128, activation=tf.nn.relu),  # regular layer (often Dense is used)
    tf.keras.layers.Dense(128, activation=tf.nn.relu),
    tf.keras.layers.Dense(16, activation=tf.nn.relu)
])
model.compile(optimizer='adam',
              loss='mean_squared_error',
              metrics=['accuracy'])

model.fit(training_data_input, training_data_target, epochs=5)
print("training done")
model.evaluate(training_data_input, training_data_target)



# write back to file
'''
mid2 = mido.MidiFile()
for trk in trks:
    # print(trk)
    track = mido.MidiTrack()
    mid2.tracks.append(track)
    note = 0
    hold = 1
    for bar in trk:
        # print(bar)
        for nt in bar:
            if note is 0:
                note = nt
            elif nt is not 129:
                note_hold_time = hold * ticks_per_sixteenth * 4
                track.append(mido.Message('note_on', note=note, velocity=100, time=note_hold_time))
                track.append(mido.Message('note_off', note=note, velocity=0, time=0))
                note = nt
                hold = 1
            else:
                hold += 1
mid2.save('reconstructed.mid')
'''





# use to round to certain number. i.e. 480 if this is one beat
#def myround(x, base=5):
#    return base * round(x/base)

# scale data for correct representation (note / 129)
# transform round(output*129)


# dunno what this is
#print(mido.format_as_string(msg, True))
#m = Message(msg)
#print(m.note)
#msg_fileds = msg.split(" ")
#print(msg_fileds)

'''
        print(msg.type)
        if msg.type == 'note_on':
            print("note on", msg.note, msg.time)
        if msg.type == 'note_off':
            print("note off", msg.time)
'''
