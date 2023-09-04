#!/usr/bin/env python3
"""This module contains the class Dataset"""
import tensorflow as tf
import tensorflow_datasets as tfds


class Dataset:
    """loads and preps a dataset for machine translation"""

    def __init__(self):
        """Class constructor"""
        self.data_train, self.data_valid = tfds.load(
            'ted_hrlr_translate/pt_to_en',
            split=['train', 'test'],
            as_supervised=True,)
        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(
            self.data_train)

    def tokenize_dataset(self, data):
        """creates sub-word tokenizers for our dataset

        Args:
        - data is a tf.data.Dataset whose examples are formatted as a tuple
          (pt, en)
            * pt is the tf.Tensor containing the Portuguese sentence
            * en is the tf.Tensor containing the corresponding English sentence

        Returns: tokenizer_pt, tokenizer_en
            - tokenizer_pt is the Portuguese tokenizer
            - tokenizer_en is the English tokenizer
        """

        tokenizer_pt = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
            (pt.numpy() for pt, _ in data), target_vocab_size=2**15)

        tokenizer_en = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
            (en.numpy() for _, en in data), target_vocab_size=2**15)

        return tokenizer_pt, tokenizer_en
