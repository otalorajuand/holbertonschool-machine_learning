#!/usr/bin/env python3
"""This module contains a script that displays the first launch of SpaceX"""
import requests

if __name__ == '__main__':
    launches_request = requests.get(
        'https://api.spacexdata.com/v4/launches/upcoming')

    launches_request_json = launches_request.json()

    first_launch_date = float('inf')
    first_launch = {}
    for launch in launches_request_json:
        if launch.get('date_unix') < first_launch_date:
            first_launch_date = int(launch.get('date_unix'))
            first_launch = launch

    # launch
    launch_name = first_launch.get('name')
    date = first_launch.get('date_local')

    # rocket info
    rocket_id = first_launch.get('rocket')
    rocket_request = requests.get(
        'https://api.spacexdata.com/v4/rockets/' +
        rocket_id)
    rocker_name = rocket_request.json().get('name')

    # launchpad info
    launch_pad_id = first_launch.get('launchpad')
    launch_pad_request = requests.get(
        'https://api.spacexdata.com/v4/launchpads/' +
        launch_pad_id)
    launch_pad_name = launch_pad_request.json().get('name')
    launch_pad_locality = launch_pad_request.json().get('locality')

    print("{} ({}) {} - {} ({})".format(launch_name, date,
                                        rocker_name, 
                                        launch_pad_name, 
                                        launch_pad_locality))
