from __future__ import division




import numpy as np
from numpy.random import seed, randint
from scipy.io import wavfile
from sklearn import svm
import linecache

from keras.models import Model
from keras.layers import GlobalAveragePooling2D
from vggish import VGGish
from preprocess_sound import preprocess_sound


def loading_data(files, sound_extractor):


    lines = linecache.getlines(files)
    sample_num = len(lines)
    seg_num- = int(1419/1.5)
    seg_len = 1.5  # 5s
    data = np.zeros((seg_num * sample_num, 496, 64, 1))

    for i in range(len(lines)):
        sound_file = lines[i][:-1]
        sr, wav_data = wavfile.read(sound_file)

        length = int(sr * seg_len)           # 5s segment


        for j in range(seg_num):
            cur_wav = wav_data[j*length:(j+1)*length]
            cur_wav = cur_wav / 32768.0
            cur_spectro = preprocess_sound(cur_wav, sr)
            cur_spectro = np.expand_dims(cur_spectro, 3)
            if cur_spectro.shape[0] == 0:
                continue

            data[i * seg_num + j, :, :, :] = cur_spectro
    print(data.shape)

    data = sound_extractor.predict(data)
    for a in data:
        print(a.shape)

    return data


if __name__ == '__main__':
    layers = ["conv1", "pool1", "conv2", "pool2", "conv3/conv3_1", "conv3/conv3_2", "pool3", "conv4/conv4_1", "conv4/conv4_2", "pool4"]
    new_layers = []
    sound_model = VGGish(include_top=False, load_weights=True)
    num = 1
    for name in layers:
        x = sound_model.get_layer(name=name).output
        output_layer = GlobalAveragePooling2D()(x)
        new_layers.append(output_layer)
#        output_layer = GlobalAveragePooling2D()(x)
#    for name in layers:
    sound_extractor = Model(input=sound_model.input, output=new_layers)
    print(sound_extractor.summary())
    # load training data
    print("loading data...")
    training_file = '/home/brain/Documents/git/VGGish/demo.txt'
    training_data2 = loading_data(training_file, sound_extractor)
    for training_data,name in zip(training_data2,layers):
        print(training_data[0].shape, len(training_data))
        data_final = np.empty((0,training_data[0].shape[0]))
        for i in range(0, len(training_data)):
            data_final = np.vstack([data_final, training_data[i]])
        print(data_final.shape, name)
        np.save("/home/brain/Documents/vggishsherlock_"+str(num)+".npy", data_final)

        num+=1
