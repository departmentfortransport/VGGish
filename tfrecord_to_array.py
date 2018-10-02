import tensorflow as tf
import numpy as np

'''
Able to extract the context, but not the actual embeddings.
Further, the embeddings are PCA'ed and quantisized - won't be compatible
with the Keras model.
https://stackoverflow.com/questions/46204992/how-can-i-extract-the-audio-embeddings-features-from-google-s-audioset
'''

def readTfRecordSamples(tfrecords_filename):

    record_iterator = tf.python_io.tf_record_iterator(path=tfrecords_filename)

    for c, string_record in enumerate(record_iterator):
        if c==0:
            example = tf.train.SequenceExample()
            example.ParseFromString(string_record)
            print(example)  # this prints the abovementioned 4 keys but NOT audio_embeddings

            # the first label can be then parsed like this:
            label = (example.context.feature['labels'].int64_list.value[0])
            print('label 1: ' + str(label))

            # this, however, does not work:
            audio_embedding = (example.feature_lists.feature_list['audio_embedding'].bytes_list.value[0])
            print(type(audio_embedding))
            print(audio_embedding.feature_list['audio_embedding'])
readTfRecordSamples('audioset_v1_embeddings/unbal_train/_0.tfrecord' )
