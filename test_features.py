import numpy as np
from keras.models import Model
from scipy.io import wavfile

from vggish import vggish
from vggish.preprocess_sound import preprocess_sound

SEG_LEN = 5  # 5s


def get_vggish():
    """ return inception3 model and preprocessor """

    vggish_model = vggish.VGGish(
        weights="audioset", include_top=True)

    model = Model(inputs=vggish_model.input,
                  outputs=vggish_model.get_layer("vggish_fc2").output)

    return model, preprocess_sound


def main():

    model, preprocessor = get_vggish()
    model.summary()

    sound_file = "/home/andrei/Downloads/output.wav"
    sr, wav_data = wavfile.read(sound_file)

    print(sr)
    print(wav_data.shape)

    length = sr * SEG_LEN

    print(length)

    cur_wav = wav_data[0:length]
    cur_spectro = preprocess_sound(cur_wav, sr)
    cur_wav = cur_wav / 32768.0
    print(cur_spectro.shape)
    cur_spectro = np.expand_dims(cur_spectro, 3)
    print(cur_spectro.shape)

    result = model.predict(cur_spectro)

    # print(result)

    print(result.shape)

    result = np.sum(result, axis=0)

    print(result)
    print(result.shape)

    # model.load_weights("/home/andrei/Downloads/test_vggish/vggish2Keras-master/vggish_weights.ckpt")


if __name__ == "__main__":
    main()
