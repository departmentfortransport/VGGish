# VGGish: A VGG-like audio classification model

This repository provides a VGGish model, implemented in Keras with tensorflow backend. This repository is developed
based on the model for [AudioSet](https://research.google.com/audioset/index.html).
For more details, please visit the [slim version](https://github.com/tensorflow/models/tree/master/research/audioset).



## Install

Create env if it is not created.

```
virtualenv -p python3 ~/envs/vggish
```

Activate env
```
source  ~/envs/vggish/bin/activate
```

Install req

```
pip install -r requirements.txt
```


## Pretrained weights in Keras h5py:

* [Model](https://drive.google.com/open?id=1mhqXZ8CANgHyepum7N4yrjiyIg6qaMe6) with the top fully connected layers

* [Model](https://drive.google.com/open?id=16JrWEedwaZFVZYvn1woPKCuWx85Ghzkp) without the top fully connected layers

## To progress with the work:

* Use `download_audioset.ipynb` to download required audio files from AudioSet
* Use `get_embeddings.py` to obtain the embeddings for those audio files using `VGGish`
* Use `vggish/classifier.py` to train a classifier based on those embeddings whether 'a car is passing by'
* Still to be done: a pipeline that takes in a new audio file and passes it through `VGGish` and the `classifier`

## Reference:

* Gemmeke, J. et. al.,
  [AudioSet: An ontology and human-labelled dataset for audio events](https://research.google.com/pubs/pub45857.html),
  ICASSP 2017

* Hershey, S. et. al.,
  [CNN Architectures for Large-Scale Audio Classification](https://research.google.com/pubs/pub45611.html),
  ICASSP 2017
