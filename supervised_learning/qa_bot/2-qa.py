#!/usr/bin/env python3
"""This module contains the function answer_loop"""
question_answer = __import__('0-qa').question_answer


def answer_loop(reference):
    """answers questions from a reference text

    Args:
    - reference is the reference text
    """

    while (1):
        user_input = input("Q: ")
        user_input = user_input.lower()
        if user_input in ['exit', 'quit', 'goodbye', 'bye']:
            print("A: Goodbye")
            break

        if question_answer(user_input, reference) is None:
            print("Sorry, I do not understand your question.")
        else:
            print("A:", question_answer(user_input, reference))

