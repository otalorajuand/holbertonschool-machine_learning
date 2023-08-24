#!/usr/bin/env python3
"""This module contains the function uni_bleu"""
import numpy as np


def uni_bleu(references, sentence):
    """
    sentence_blue
    Args:
        references: list of reference translations
        sentence: list containing the model proposed sentence
    Returns: unigram BLEU score
    """

    len_sen = len(sentence)

    word_references = list(
        set([x for reference in references for x in reference]))

    counts = 0
    for word in sentence:
        if word in word_references:
            counts += 1

    min_reference = min([len(reference) for reference in references])

    bp = 1 if len_sen > min_reference else np.exp(
        1 - (min_reference / len_sen))

    return bp * (counts / len_sen)
