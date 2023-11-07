#!/usr/bin/env python3
"""This module includes the script that retrieves github info"""
import requests
import sys
import time


if __name__ == '__main__':

    if len(sys.argv) != 2:
        raise ValueError('Incorrect number of input parameters')

    r = requests.get(sys.argv[-1])
    r_status_code = r.status_code

    if r_status_code == 200:
        print(r.json().get('location'))
    elif r_status_code == 403:
        rate_limit = r.headers.get('X-Ratelimit-Reset')
        time_minutes = round((int(rate_limit) - time.time()) / 60)
        print('Reset in {} min'.format(time_minutes))
    elif r.status_code == 404:
        print('Not found')
