import numpy as np
from numpy import genfromtxt
from keras.models import Model
from scipy.io import wavfile

import os.path
from os import listdir
from os.path import isfile, join

from vggish import vggish
from vggish.preprocess_sound import preprocess_sound

import time

SEG_LEN = 5  # 5s
#TODO: find out why would you want to cut the SEG_LEN


def get_vggish():
    """ return inception3 model and preprocessor """

    vggish_model = vggish.VGGish(
        weights="audioset", include_top=True)

    model = Model(inputs=vggish_model.input,
                  outputs=vggish_model.get_layer("vggish_fc2").output)

    return model


def get_embeddings(model, sound_file):
    '''
    Obtain the (128,) embeddings for a given sound file by
    passing through VGGish
    '''

    sr, wav_data = wavfile.read(sound_file)
    # print(wav_data.shape)
    # print(np.max(wav_data[:,0]))
    wav_data = wav_data / 32768.0  # to force within [-1,1]

    # print('sound rate: ', sr)
    # print('wav_data.shape: ', wav_data.shape)
    # length = sr * SEG_LEN
    # print(length)
    # cur_wav = wav_data[0:length]
    # cur_spectro = preprocess_sound(cur_wav, sr)

    cur_spectro = preprocess_sound(wav_data, sr)  # take a while
    # print('cur_spectro.shape: ', cur_spectro.shape)
    cur_spectro = np.expand_dims(cur_spectro, 3)  # prediction doesn't work o/w
    # print('cur_spectro.shape: ', cur_spectro.shape)

    result = model.predict(cur_spectro)  # take a while

    # print(result)
    # print('result shape: ', result.shape)

    result = np.sum(result, axis=0)  # need to experiment with this

    # print('result: ', result)
    # print('result shape: ', result.shape)

    return result

# check one example:
# model = get_vggish()
# result = get_embeddings(model, "/Users/liucija/repos/VGGish/data/audio/unbal/car_passing_by/_-xwxCG8-iQ_30000_40000.wav")
# result.shape
# result

def dir_files(path):
    '''
    Lists files in a directory, ignoring hidden files
    '''
    return [f for f in listdir(path) if isfile(join(path, f)) and not f.startswith('.')]


def create_emb_data(dataset, label):
    '''
    Gets embeddings from audio files which are stored in
    'data/audio/dataset/label/' for given dataset and label.
    Files must already be downloaded and stored in this folder.
    The embeddings are written to csv for individual file to save memory.

    Keywords
    --------
    dataset: 'train', 'eval' or 'unbal'
    label: any folder name that exists in data/audio/dataset
    '''
    directory = 'data/audio/'+dataset+'/'+label+'/'
    files = dir_files(directory)
    # files = files[:2]  # limited when downloading
    model = get_vggish()  # takes a while
    # model.summary()

    for c, f in enumerate(files):
        print('Processing {} file out of {}'.format(c+1, len(files)))
        embs = get_embeddings(model, directory+f)
        newpath = 'data/embs/'+dataset+'/'+label+'/'
        if not os.path.exists(newpath):
            os.makedirs(newpath)
        np.savetxt(newpath+'emb_'+str(c)+".csv", embs, delimiter=",")


def combine_embs(dataset, label):
    '''
    Takes a name of the folder and combines embeddings available in
    'data/embs/'

    Keywords
    --------
    dataset: 'train', 'eval' or 'unbal'
    label: any folder name that exists in data/embs/dataset
    '''
    directory = 'data/embs/'+dataset+'/'+label+'/'
    files = dir_files(directory)
    X = np.zeros((len(files), 128))

    for c, f in enumerate(files):
        print('Processing emb {} {}'.format(label, c+1))
        embs = genfromtxt(directory+f, delimiter=',')
        X[c] = embs

    np.savetxt('data/embs/'+dataset+'/'+label+'_embs.csv', X, delimiter=",")

    return X


def make_dataset(dataset, pos_emb, neg_emb):
    '''
    '''

    X = np.append(pos_emb, neg_emb, 0)
    print('Final shape: {}'.format(X.shape))
    print(X[0][0] == pos_emb[0][0])
    print(X[-1][0] == neg_emb[-1][0])
    print('Saving X to .csv')
    np.savetxt("data/embs/X_"+dataset+".csv", X, delimiter=",")

    y = np.append(np.ones(len(pos_emb)), np.zeros(len(neg_emb)))
    print('Saving y to .csv')
    np.savetxt("data/embs/y_"+dataset+".csv", y, delimiter=",")

    return X, y


dataset = 'unbal'
create_emb_data(dataset, 'car_passing_by')
create_emb_data(dataset, 'outside_rural')

car_embs = combine_embs(dataset, 'car_passing_by')
outside_embs = combine_embs(dataset, 'outside_rural')

X, y = make_dataset(dataset, car_embs, outside_embs)

#TODO: cleanup the multiple unnecessary file savings
