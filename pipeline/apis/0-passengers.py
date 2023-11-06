#!/usr/bin/env python3
"""This module contains the function availableShips"""
import requests


def availableShips(passengerCount):
    """Returns the list of ships that can hold a given number of passengers
    """
    url = 'https://swapi-api.hbtn.io/api/starships'
    r = requests.get(url)
    next_ = r.json().get('next')

    ships = []
    while next_:

        lista = r.json().get("results")

        for ship in lista:
            try:
                aux = ship.get('passengers').replace(',', '')
                n_passengers = int(aux)
            except ValueError:
                n_passengers = -1

            if n_passengers >= passengerCount:
                ships.append(ship['name'])

        r = requests.get(next_)
        next_ = r.json().get('next')

    return ships
