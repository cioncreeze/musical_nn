import mido
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import my_helpers as mh


#print("tensorflow version: ", tf.__version__)

# MIDI_sample.mid

mid = mido.MidiFile('./inputs/inp_1_cMaj_4-4th_temp_1.mid')

# midi tempo is microseconds per beat. standard is 500000
# message time attribute is number of ticks to wait
print(mid.ticks_per_beat) # ticks are midi time unit. if tempo is 500000 and ticks per beat is 480 then there are ~1042
# microseconds per tick
ticks_per_sixteenth = round(mid.ticks_per_beat / 4)
trks = []

# read in file and transform to nn representation
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

    # for bar in bars:
    #     print(bar, len(bar))



    trks.append(bars)

training_track1 = mh.extract_bars('./inputs/inp_1_cMaj_4-4th_temp_1.mid')
training_track2 = mh.extract_bars('./inputs/inp_2_cMaj_4-4th_temp_1.mid')
training_track3 = mh.extract_bars('./inputs/inp_3_cMaj_4-4th_temp_1.mid')
training_track4 = mh.extract_bars('./inputs/inp_4_cMaj_4-4th_temp_1.mid')
training_track5 = mh.extract_bars('./inputs/inp_5_cMaj_4-4th_temp_1.mid')

# print(len(bars))
training_track = training_track1 + training_track2 + training_track3 + training_track4 + training_track5
training_data_input, training_data_target = mh.generate_training_data(training_track)
# generate training data
#training_data_input = []
#training_data_target = []
# just take first track here for convenience sake
#training_track = trks[0] # hardcoded for now, later take first non empty track
#for i in range(len(training_track)-2):
    #print(training_track[i], training_track[i+1], training_track[i+2])
#    training_data_input.append(training_track[i] + training_track[i+1])
#    training_data_target.append(training_track[i+2])

#training_data_input = np.array(training_data_input)
#training_data_target = np.array(training_data_target)

# create function returning standardized model
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

def write_to_file(bars,ticks_per_beat, ticks_per_sixteenth, filepath="./", filename="reconstructed.mid"):
    mid = mido.MidiFile(ticks_per_beat=ticks_per_beat)
    track = mido.MidiTrack()
    mid.tracks.append(track)
    note = -1
    hold = 1
    for bar in bars:
        # print(bar)
        for nt in bar:
            if note is -1:
                note = nt
            elif nt is not 129:
                note_hold_time = hold * ticks_per_sixteenth #* 4
                track.append(mido.Message('note_on', note=note, velocity=100, time=note_hold_time))
                track.append(mido.Message('note_off', note=note, velocity=0, time=0))
                note = nt
                hold = 1
            else:
                hold = hold + 1
    mid.save(filepath + filename)

# make the model
model = mh.make_model()

# train for one epoch
model.fit(training_data_input, training_data_target, epochs=1)
model.save('./one_epoch_results/one_epoch.h5')
out_bars = generate_more_music(model, training_data_input[-1], 64)
mh.write_to_file(out_bars, mid.ticks_per_beat, filepath="./one_epoch_results/", filename="out_1")
loss, acc = model.evaluate(training_data_input,  training_data_target, verbose=2)
mh.log_evaluation(loss, acc, "1")

# train for 4 more (5 total)
model.fit(training_data_input, training_data_target, epochs=4)
model.save('./five_epoch_results/five_epochs.h5')
out_bars = generate_more_music(model, training_data_input[-1], 64)
mh.write_to_file(out_bars, mid.ticks_per_beat, filepath="./five_epoch_results/", filename="out_5")
loss, acc = model.evaluate(training_data_input,  training_data_target, verbose=2)
mh.log_evaluation(loss, acc, "5")

# train for 5 more (10 total)
model.fit(training_data_input, training_data_target, epochs=5)
model.save('./ten_epoch_results/ten_epochs.h5')
out_bars = generate_more_music(model, training_data_input[-1], 64)
mh.write_to_file(out_bars, mid.ticks_per_beat, filepath="./ten_epoch_results/", filename="out_10")
loss, acc = model.evaluate(training_data_input,  training_data_target, verbose=2)
mh.log_evaluation(loss, acc, "10")

# train for 10 more (20 total)
model.fit(training_data_input, training_data_target, epochs=10)
model.save('./twenty_epoch_results/twenty_epochs.h5')
out_bars = generate_more_music(model, training_data_input[-1], 64)
mh.write_to_file(out_bars, mid.ticks_per_beat, filepath="./twenty_epoch_results/", filename="out_20")
loss, acc = model.evaluate(training_data_input,  training_data_target, verbose=2)
mh.log_evaluation(loss, acc, "20")


#print(len(training_data_input), np.shape(training_data_input))
#print(training_data_input)
#print(training_data_input[-1])
#print(np.shape(training_data_input[-1]))
#print(np.shape(training_data_input[-1].T))
#start_point = training_data_input[-1]
#start_point = start_point.reshape(32,1).T
#result = model.predict(start_point)
#print(np.around(result))
#print(type(result))
#print("training done")
#model.evaluate(training_data_input, training_data_target)






# write back to file

#def write_to_file(path, info...):
    # TODO

mid2 = mido.MidiFile(ticks_per_beat=mid.ticks_per_beat)
for trk in trks:
    # print(trk)
    track = mido.MidiTrack()
    mid2.tracks.append(track)
    note = -1
    hold = 1
    for bar in trk:
        # print(bar)
        for nt in bar:
            if note is -1:
                note = nt
            elif nt is not 129:
                note_hold_time = hold * ticks_per_sixteenth #* 4
                track.append(mido.Message('note_on', note=note, velocity=100, time=note_hold_time))
                track.append(mido.Message('note_off', note=note, velocity=0, time=0))
                note = nt
                hold = 1
            else:
                hold = hold + 1
mid2.save('reconstructed.mid')






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
