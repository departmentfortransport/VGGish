import numpy as np
from keras.models import Model
from scipy.io import wavfile

from os import listdir
from os.path import isfile, join

from vggish import vggish
from vggish.preprocess_sound import preprocess_sound

SEG_LEN = 5  # 5s
#TODO: find out why would you want to cut the SEG_LEN


def get_vggish():
    """ return inception3 model and preprocessor """

    vggish_model = vggish.VGGish(
        weights="audioset", include_top=True)

    model = Model(inputs=vggish_model.input,
                  outputs=vggish_model.get_layer("vggish_fc2").output)

    return model, preprocess_sound


def get_embeddings(sound_file):
    '''
    Obtain the (128,) embeddings for a given sound file by
    passing through VGGish
    '''

    model, preprocessor = get_vggish()
    # model.summary()

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

    cur_spectro = preprocess_sound(wav_data, sr)
    # print('cur_spectro.shape: ', cur_spectro.shape)
    cur_spectro = np.expand_dims(cur_spectro, 3)  # prediction doesn't work o/w
    # print('cur_spectro.shape: ', cur_spectro.shape)

    result = model.predict(cur_spectro)

    # print(result)
    # print('result shape: ', result.shape)

    result = np.sum(result, axis=0)

    # print('result: ', result)
    # print('result shape: ', result.shape)

    return result

# check one example:
# result = get_embeddings("/Users/liucija/repos/VGGish/data/audio/eval/car/-CZ1LIc8aos_20000_30000.wav")
# result.shape
# result

def dir_files(path):
    '''
    Lists files in a directory, ignoring hidden files
    '''
    return [f for f in listdir(path) if isfile(join(path, f)) and not f.startswith('.')]


def create_emb_data(directory):
    '''
    Gets embeddings and puts them in one array for all files
    '''
    files = dir_files(directory)
    # files = files[:2]  # limited when downloading
    X = np.zeros((len(files), 128))
    for c, f in enumerate(files):
        print('Processing {} file out of {}'.format(c+1, len(files)))
        embs = get_embeddings(directory+f)
        X[c] = embs

    return X


def combine_data(dataset):
    '''
    Obtains embeddings for thhe positive and negative class and puts them
    into one array
    ----
    Keywords
    - dataset: 'train' or 'eval'
    '''
    car_embs = create_emb_data('data/audio/'+dataset+'/car/')
    print('Car data shape: {}'.format(car_embs.shape))
    print('Saving to .csv')
    np.savetxt("data/embs/car_embs_"+dataset+".csv", car_embs, delimiter=",")

    outdoor_embs = create_emb_data('data/audio/'+dataset+'/outside_rural/')
    print('Outdoor data shape: {}'.format(outdoor_embs.shape))
    print('Saving to .csv')
    np.savetxt("data/embs/outdoor_embs_"+dataset+".csv", outdoor_embs, delimiter=",")

    X = np.append(car_embs, outdoor_embs, 0)
    print('Final shape: {}'.format(X.shape))
    print(X[0][0] == car_embs[0][0])
    print(X[-1][0] == outdoor_embs[-1][0])
    print('Saving to .csv')
    np.savetxt("data/embs/X_"+dataset+".csv", X, delimiter=",")

    y = np.append(np.ones(len(car_embs)), np.zeros(len(outdoor_embs)))
    print('Saving to .csv')
    np.savetxt("data/embs/y_"+dataset+".csv", y, delimiter=",")

    return X, y

# Currently this function just stops for some reason on outdoor,
# so broke it down into parts below

X_train, y_train = combine_data('train')
X_eval, y_eval = combine_data('eval')

# Broken down in steps:

dataset = 'eval'  # then run for 'train'

car_embs = create_emb_data('data/audio/'+dataset+'/car/')
print('Car data shape: {}'.format(car_embs.shape))
print('Saving to .csv')
np.savetxt("data/embs/car_embs_"+dataset+".csv", car_embs, delimiter=",")

outdoor_embs = create_emb_data('data/audio/'+dataset+'/outside_rural/')
print('Outdoor data shape: {}'.format(outdoor_embs.shape))
print('Saving to .csv')
np.savetxt("data/embs/outdoor_embs_"+dataset+".csv", outdoor_embs, delimiter=",")

X = np.append(car_embs, outdoor_embs, 0)
print(X[0][0] == car_embs[0][0])
print(X[-1][0] == outdoor_embs[-1][0])
print('Final shape: {}'.format(X.shape))
print('Saving X to .csv')
np.savetxt("data/embs/X_"+dataset+".csv", X, delimiter=",")

y = np.append(np.ones(len(car_embs)), np.zeros(len(outdoor_embs)))
print('Saving y to .csv')
np.savetxt("data/embs/y_"+dataset+".csv", y, delimiter=",")
