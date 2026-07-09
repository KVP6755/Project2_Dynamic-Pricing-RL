"""
Project 2: Dynamic Pricing using Reinforcement Learning

Member 3: Market Simulation
Implements inventory management, customer demand simulation,
revenue calculation, and booking probability logic.

Author  : Varnika Valliammai V
File    : market_simulation.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# 1. Inventory Management


class InventoryManager:
    """
    Manages hotel room inventory over a booking period.

    Tracks how many rooms are available, handles bookings,
    and prevents overbooking.
    """

    def __init__(self, total_rooms=100, total_days=30):
        """
        Parameters:
            total_rooms : total rooms available in the hotel
            total_days  : number of days in the booking window
        """
        self.total_rooms  = total_rooms
        self.total_days   = total_days
        self.rooms_left   = total_rooms
        self.day          = 0
        self.booking_log  = []

    def reset(self):
        """Resets inventory to start a new episode."""
        self.rooms_left  = self.total_rooms
        self.day         = 0
        self.booking_log = []

    def book_rooms(self, num_rooms):
        """
        Books a number of rooms if available.

        Parameters:
            num_rooms : how many rooms a customer wants to book

        Returns:
            booked : actual number of rooms booked (0 if sold out)
        """
        booked = min(num_rooms, self.rooms_left)
        self.rooms_left -= booked
        self.booking_log.append({
            'day'        : self.day,
            'rooms_booked': booked,
            'rooms_left' : self.rooms_left
        })
        return booked

    def advance_day(self):
        """Moves to the next day."""
        self.day += 1

    def is_sold_out(self):
        """Returns True if no rooms are left."""
        return self.rooms_left == 0

    def occupancy_rate(self):
        """Returns what percentage of rooms are booked."""
        return (self.total_rooms - self.rooms_left) / self.total_rooms

    def summary(self):
        """Prints inventory summary."""
        print(f"Day          : {self.day}")
        print(f"Rooms left   : {self.rooms_left}/{self.total_rooms}")
        print(f"Occupancy    : {self.occupancy_rate()*100:.1f}%")
