"""
Project 2: Travel & Hospitality - Dynamic Pricing using Reinforcement Learning (RL)

Week 4: Policy Evaluation & Business Metrics Analysis

Member 2: Business Metrics Analysis

Implements business performance metrics for evaluating the Dynamic Pricing
strategy. The module calculates key performance indicators (KPIs) such as
Total Revenue, Average Revenue per Booking, Booking Success Rate, and
Occupancy Rate. It also compares the performance of the Baseline pricing
strategy with the DQN-based pricing strategy.

Author  : Shilpa S Nair
File    : business_metrics.py
"""



import numpy as np

# Create BusinessMetrics class.
class BusinessMetrics:

    def __init__(self):

        # Sample booking prices
        self.prices = [
            4000,
            4500,
            5000,
            5500,
            6000
        ]

        # Sample successful bookings
        self.bookings = [
            18,
            25,
            30,
            20,
            12
        ]

        # Total booking requests
        self.total_requests = [
        20,
        30,
        35,
        25,
        15
        ]
        print(self.__dict__)

    # --------------------------------------------------------
    # Calculate Total Revenue
    # --------------------------------------------------------
    def calculate_total_revenue(self):

        revenue = 0

        for price, booking in zip(
            self.prices,
            self.bookings
        ):

            revenue += price * booking

        return revenue


    # --------------------------------------------------------
    # Day 2
    # Calculate Average Revenue per Booking
    # --------------------------------------------------------
    def calculate_average_revenue(self):

    # Calculate total revenue
        total_revenue = self.calculate_total_revenue()

    # Calculate total successful bookings
        total_bookings = sum(self.bookings)

    # Avoid division by zero
        if total_bookings == 0:
            return 0

    # Calculate average revenue
        average_revenue = total_revenue / total_bookings

        return average_revenue

    # --------------------------------------------------------
    # Calculate Booking Success Rate
    # --------------------------------------------------------

    def calculate_booking_success_rate(self):

    # Total successful bookings
        successful_bookings = sum(self.bookings)

    # Total booking requests
        total_requests = sum(self.total_requests)

    # Avoid division by zero
        if total_requests == 0:
            return 0

    # Calculate success rate
        success_rate = (
            successful_bookings / total_requests
        ) * 100

        return success_rate


    # --------------------------------------------------------
# Calculate Occupancy Rate
# --------------------------------------------------------

    def calculate_occupancy_rate(self):

    # Total available rooms
        total_capacity = 120

    # Total occupied rooms
        occupied_rooms = sum(self.bookings)

    # Avoid division by zero
        if total_capacity == 0:
         return 0

        occupancy_rate = (
            occupied_rooms / total_capacity
        ) * 100

        return occupancy_rate