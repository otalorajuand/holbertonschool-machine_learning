#!/usr/bin/env python3
"""This module includes the function sentientPlanets"""
import requests


def sentientPlanets():
    """Returns the list of names of the home planets of all sentient species
    """

    url = 'https://swapi-api.hbtn.io/api/species'
    r = requests.get(url)
    next_ = r.json().get('next')

    planets = []
    while True:

        lista = r.json().get("results")

        for species in lista:
            homeworld = species.get('homeworld')

            if homeworld:
                planet_request = requests.get(homeworld)
                if species.get('classification') == 'sentient' or species.get(
                        'designation') == 'sentient':
                    planets.append(planet_request.json().get('name'))

        if next_ is None:
            break

        r = requests.get(next_)
        next_ = r.json().get('next')

    return planets
