#!/usr/bin/env python3
"""This module contains a script that displays the first launch of SpaceX"""
import requests

if __name__ == '__main__':
    r = requests.get('https://api.spacexdata.com/v5/launches/latest')

    request_json = r.json()

    # launch
    launch_name = request_json.get('name')
    date = request_json.get('date_local')


    # rocket info
    rocket_id = request_json.get('rocket')
    rocket_request = requests.get(
        'https://api.spacexdata.com/v4/rockets/' +
        rocket_id)
    rocker_name = rocket_request.json().get('name')

    # launchpad info
    launch_pad_id = request_json.get('launchpad')
    launch_pad_request = requests.get(
        'https://api.spacexdata.com/v4/launchpads/' +
        launch_pad_id)
    launch_pad_name = launch_pad_request.json().get('name')
    launch_pad_locality = launch_pad_request.json().get('locality')


    print("{} ({}) {} - {} ({})".format(launch_name, date,
        rocker_name, launch_pad_name, launch_pad_locality))
