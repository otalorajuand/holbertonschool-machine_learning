#!/usr/bin/env python3
"""This module contains the function answer_loop"""
single_question_answer = __import__('0-qa').question_answer
semantic_search = __import__('3-semantic_search').semantic_search


def question_answer(coprus_path):
    """answers questions from multiple reference texts

    Args:
    - corpus_path is the path to the corpus of reference documents
    """

    while (1):
        user_input = input("Q: ")
        user_input = user_input.lower()
        if user_input in ['exit', 'quit', 'goodbye', 'bye']:
            print("A: Goodbye")
            break

        reference = semantic_search(coprus_path, user_input)
        answer = single_question_answer(user_input, reference)
        if answer is None:
            print("Sorry, I do not understand your question.")
        else:
            print("A:", answer)
