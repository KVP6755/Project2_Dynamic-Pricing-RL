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


# 2. Customer Demand Simulation

class DemandSimulator:
    """
    Simulates how many customers arrive each day and
    how many rooms they want to book.

    Demand is influenced by:
    - Base demand (average customers per day)
    - Day of week (weekends busier)
    - Season (peak/off-peak)
    - Random variation (real-world unpredictability)
    """

    def __init__(self, base_demand=10, random_state=42):
        """
        Parameters:
            base_demand  : average number of customer groups per day
            random_state : seed for reproducibility
        """
        self.base_demand  = base_demand
        self.rng          = np.random.RandomState(random_state)

    def get_daily_demand(self, day, season='normal'):
        """
        Simulates customer arrivals for a given day.

        Parameters:
            day    : current day (0-29)
            season : 'peak', 'normal', or 'off_peak'

        Returns:
            num_customers : number of customer groups arriving today
        """
        # Season multiplier
        season_multiplier = {
            'peak'    : 1.5,
            'normal'  : 1.0,
            'off_peak': 0.6
        }.get(season, 1.0)

        # Weekend effect (days 5,6,12,13... are weekends)
        weekend_boost = 1.3 if (day % 7) in [5, 6] else 1.0

        # Final demand with random variation
        mean_demand = self.base_demand * season_multiplier * weekend_boost
        num_customers = int(self.rng.poisson(mean_demand))

        return max(0, num_customers)

    def get_rooms_requested(self, num_customers):
        """
        For each customer group, randomly decide how many
        rooms they want (1-3 rooms per group).

        Parameters:
            num_customers : number of customer groups

        Returns:
            total_rooms_requested : total rooms all customers want
        """
        if num_customers == 0:
            return 0

        rooms_per_group = self.rng.randint(1, 4, size=num_customers)
        return int(rooms_per_group.sum())

# 3. Booking Probability Logic
def booking_probability(price, base_price=100.0, sensitivity=0.02):
    """
    Calculates the probability that a customer will book
    at a given price.

    Logic: higher price = lower booking probability.
    Uses an exponential decay model.

    Parameters:
        price       : current room price being offered
        base_price  : reference price (probability = 0.8 at base)
        sensitivity : how sensitive customers are to price changes
                      (higher = more price-sensitive)

    Returns:
        probability : float between 0 and 1
    """
    probability = 0.8 * np.exp(-sensitivity * (price - base_price))
    return float(np.clip(probability, 0.0, 1.0))


def will_customer_book(price, base_price=100.0,
                       sensitivity=0.02, rng=None):
    """
    Randomly decides if a customer books based on
    the booking probability at the current price.

    Parameters:
        price       : current room price
        base_price  : reference price
        sensitivity : price sensitivity
        rng         : numpy RandomState for reproducibility

    Returns:
        True if customer books, False otherwise
    """
    prob = booking_probability(price, base_price, sensitivity)
    if rng is None:
        return np.random.random() < prob
    return rng.random() < prob

# 4. Revenue Calculation

def calculate_revenue(rooms_booked, price):
    """
    Calculates revenue from a single booking event.

    Parameters:
        rooms_booked : number of rooms successfully booked
        price        : price per room

    Returns:
        revenue : total revenue from this booking
    """
    return rooms_booked * price


def run_simulation(price=100.0, total_rooms=100,
                   total_days=30, season='normal'):
    """
    Runs a full 30-day market simulation at a fixed price.

    Combines all components:
    inventory + demand + booking probability + revenue

    Parameters:
        price       : room price for this simulation run
        total_rooms : total rooms available
        total_days  : number of days to simulate
        season      : demand season ('peak', 'normal', 'off_peak')

    Returns:
        results_df  : daily simulation log as a DataFrame
        total_revenue: total revenue earned over all days
    """
    inventory = InventoryManager(total_rooms=total_rooms,
                                 total_days=total_days)
    demand    = DemandSimulator(base_demand=10, random_state=42)

    daily_log = []

    for day in range(total_days):
        if inventory.is_sold_out():
            break

        # How many customers arrive today
        num_customers = demand.get_daily_demand(day, season=season)

        # How many rooms they want
        rooms_requested = demand.get_rooms_requested(num_customers)

        # How many actually decide to book at this price
        actual_bookers = sum(
            will_customer_book(price, rng=demand.rng)
            for _ in range(num_customers)
        )

        # Rooms successfully booked
        rooms_to_book = min(rooms_requested, actual_bookers)
        rooms_booked  = inventory.book_rooms(rooms_to_book)

        # Revenue from today
        revenue = calculate_revenue(rooms_booked, price)

        daily_log.append({
            'day'            : day + 1,
            'customers'      : num_customers,
            'rooms_requested': rooms_requested,
            'rooms_booked'   : rooms_booked,
            'rooms_left'     : inventory.rooms_left,
            'price'          : price,
            'revenue'        : revenue
        })

        inventory.advance_day()

    results_df    = pd.DataFrame(daily_log)
    total_revenue = results_df['revenue'].sum()

    return results_df, total_revenue



# 5. Visualizations
def plot_simulation_results(results_df, title="Market Simulation"):
    """
    Plots daily bookings, inventory, and revenue
    from a simulation run.
    """
    fig, axes = plt.subplots(3, 1, figsize=(10, 10))

    # Plot 1: Daily bookings
    axes[0].bar(results_df['day'], results_df['rooms_booked'],
                color='steelblue')
    axes[0].set_title('Daily Rooms Booked')
    axes[0].set_xlabel('Day')
    axes[0].set_ylabel('Rooms Booked')
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Inventory remaining
    axes[1].plot(results_df['day'], results_df['rooms_left'],
                 color='orange', linewidth=2)
    axes[1].set_title('Rooms Remaining Over Time')
    axes[1].set_xlabel('Day')
    axes[1].set_ylabel('Rooms Left')
    axes[1].grid(True, alpha=0.3)

    # Plot 3: Daily revenue
    axes[2].bar(results_df['day'], results_df['revenue'],
                color='green', alpha=0.7)
    axes[2].set_title('Daily Revenue')
    axes[2].set_xlabel('Day')
    axes[2].set_ylabel('Revenue ($)')
    axes[2].grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('simulation_results.png')
    plt.show()


# 6. Run and test

if __name__ == "__main__":

    print("=" * 50)
    print("MARKET SIMULATION TEST")
    print("=" * 50)

    # Test booking probability
    print("\nBooking probability at different prices:")
    for price in [70, 100, 130, 160]:
        prob = booking_probability(price)
        print(f"  Price ${price}: {prob:.2f} probability of booking")

    # Run full simulation
    print("\nRunning 30-day simulation at $100/room...")
    results, revenue = run_simulation(price=100.0, season='normal')

    print(f"\nTotal revenue    : ${revenue:,.2f}")
    print(f"Total rooms booked: {results['rooms_booked'].sum()}")
    print(f"Days simulated   : {len(results)}")
    print("\nFirst 5 days:")
    print(results.head().to_string(index=False))

    # Plot results
    plot_simulation_results(results, title="Market Simulation — $100/room")

    print("\n✓ Market simulation complete.")
    print("  market_simulation.py ready to plug into PricingEnv")