#!/usr/bin/env python3
import tensorflow as tf
import tensorflow_datasets as tfds

train, test = tfds.load('ted_hrlr_translate/pt_to_en', split=['train', 'test'], as_supervised=True)
for pt, en in train.take(1):
  print(pt.numpy().decode('utf-8'))
  print(en.numpy().decode('utf-8'))
