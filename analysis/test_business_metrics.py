from business_metrics import BusinessMetrics

print("=" * 55)
print("DAY 3 - BOOKING SUCCESS RATE")
print("=" * 55)

metrics = BusinessMetrics()

print("\nPricing Levels")
print("-" * 30)
print(metrics.prices)

print("\nSuccessful Bookings")
print("-" * 30)
print(metrics.bookings)

print("\nBooking Requests")
print("-" * 30)
print(metrics.total_requests)

# -------------------------
# Total Revenue Calculation
# -------------------------

total_revenue = metrics.calculate_total_revenue()

print("\nTotal Revenue")
print("-" * 30)
print(f"₹ {total_revenue:,}")

# -------------------------
# Average Revenue per Booking Calculation
# -------------------------

total_bookings = sum(metrics.bookings)

print("\nTotal Successful Bookings")
print("-" * 30)
print(total_bookings)

average_revenue = metrics.calculate_average_revenue()

print("\nAverage Revenue per Booking")
print("-" * 30)
print(f"₹ {average_revenue:.2f}")

# -------------------------
# Booking Success Rate Calculation
# -------------------------

booking_success_rate = metrics.calculate_booking_success_rate()

print("\nBooking Success Rate")
print("-" * 30)
print(f"{booking_success_rate:.2f}%")


# -------------------------
# Occupancy Rate Calculation
# -------------------------

occupancy_rate = metrics.calculate_occupancy_rate()

print("\nOccupancy Rate")
print("-" * 30)
print(f"{occupancy_rate:.2f}%")

print("\n" + "=" * 55)
print("IMPLEMENTATION COMPLETED SUCCESSFULLY")
print("=" * 55)

print("✓ Total Revenue calculated")
print("✓ Average Revenue per Booking calculated")
print("✓ Booking Success Rate calculated")
print("✓ Occupancy Rate calculated")

print("\nThe BusinessMetrics module now includes all required KPI calculations for business performance evaluation.")
print("=" * 55)