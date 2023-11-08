#!/usr/bin/env python3
"""This module contains a script that displays the number of
   launches per rocket of SpaceX"""
import requests


if __name__ == '__main__':

    rockets = {}

    launches_request = requests.get(
        'https://api.spacexdata.com/v3/launches')

    launches_list = launches_request.json()

    for launch in launches_list:
        rocket = launch.get('rocket')
        rocket_name = rocket.get('rocket_name')

        if rocket_name not in rockets.keys():
            rockets[rocket_name] = 1
        else:
            rockets[rocket_name] += 1

    sorted_rockets = dict(
        sorted(
            rockets.items(),
            key=lambda x: x[1],
            reverse=True))
    for k, v in sorted_rockets.items():
        print("{}: {}".format(k, v))
