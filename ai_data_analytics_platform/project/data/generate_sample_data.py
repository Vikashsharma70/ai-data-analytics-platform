"""
generate_sample_data.py
------------------------
Generates a realistic sample_sales_data.csv (1,000+ rows) so the platform
can be tested immediately without needing to source an external dataset.

Run:  python data/generate_sample_data.py
"""

from __future__ import annotations

import random
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

random.seed(42)
np.random.seed(42)

N_RECORDS = 1200

REGIONS = ["North", "South", "East", "West", "Central"]
CITIES_BY_REGION = {
    "North": ["Delhi", "Chandigarh", "Lucknow"],
    "South": ["Bengaluru", "Chennai", "Hyderabad"],
    "East": ["Kolkata", "Patna", "Bhubaneswar"],
    "West": ["Mumbai", "Pune", "Ahmedabad"],
    "Central": ["Bhopal", "Nagpur", "Indore"],
}
CATEGORIES = {
    "Electronics": ["Smartphone", "Laptop", "Headphones", "Smartwatch", "Tablet"],
    "Furniture": ["Office Chair", "Desk", "Bookshelf", "Sofa", "Dining Table"],
    "Clothing": ["T-Shirt", "Jeans", "Jacket", "Sneakers", "Cap"],
    "Groceries": ["Rice 5kg", "Cooking Oil", "Snack Pack", "Beverages", "Spices Combo"],
    "Stationery": ["Notebook Set", "Pen Pack", "Backpack", "Desk Organizer", "Whiteboard"],
}
CUSTOMER_SEGMENTS = ["Consumer", "Corporate", "Small Business"]

BASE_PRICE = {
    "Smartphone": 18000, "Laptop": 55000, "Headphones": 2500, "Smartwatch": 4500, "Tablet": 16000,
    "Office Chair": 6500, "Desk": 9000, "Bookshelf": 4200, "Sofa": 22000, "Dining Table": 15000,
    "T-Shirt": 600, "Jeans": 1500, "Jacket": 2800, "Sneakers": 3200, "Cap": 350,
    "Rice 5kg": 450, "Cooking Oil": 320, "Snack Pack": 150, "Beverages": 90, "Spices Combo": 400,
    "Notebook Set": 250, "Pen Pack": 100, "Backpack": 1800, "Desk Organizer": 700, "Whiteboard": 1600,
}

COST_RATIO = {
    "Electronics": 0.78, "Furniture": 0.65, "Clothing": 0.55, "Groceries": 0.72, "Stationery": 0.6,
}

start_date = datetime(2023, 1, 1)
end_date = datetime(2024, 12, 31)
date_span = (end_date - start_date).days

first_names = ["Aarav", "Vivaan", "Ishaan", "Ananya", "Diya", "Aditi", "Kabir", "Meera",
               "Rohan", "Sara", "Kunal", "Priya", "Nikhil", "Neha", "Arjun", "Pooja"]
last_names = ["Sharma", "Verma", "Gupta", "Reddy", "Iyer", "Khan", "Singh", "Das",
              "Mehta", "Nair", "Patel", "Rao"]

n_customers = 260
customers = []
for i in range(1, n_customers + 1):
    name = f"{random.choice(first_names)} {random.choice(last_names)}"
    customers.append((f"CUST-{i:04d}", name, random.choice(CUSTOMER_SEGMENTS)))

rows = []
for order_num in range(1, N_RECORDS + 1):
    category = random.choice(list(CATEGORIES.keys()))
    product = random.choice(CATEGORIES[category])
    region = random.choice(REGIONS)
    city = random.choice(CITIES_BY_REGION[region])
    cust_id, cust_name, segment = random.choice(customers)

    order_date = start_date + timedelta(days=random.randint(0, date_span))
    # Add a mild seasonal boost around Oct-Dec (festive season) and slight upward yearly trend
    seasonal_boost = 1.25 if order_date.month in (10, 11, 12) else 1.0
    year_trend = 1.0 + (order_date.year - 2023) * 0.08

    quantity = random.randint(1, 8)
    unit_price = BASE_PRICE[product] * random.uniform(0.9, 1.1)
    discount = round(random.choice([0, 0, 0.05, 0.1, 0.15, 0.2, 0.3]), 2)

    gross_sales = unit_price * quantity * seasonal_boost * year_trend
    sales = round(gross_sales * (1 - discount), 2)
    cost = round(sales * COST_RATIO[category] * random.uniform(0.9, 1.1), 2)
    profit = round(sales - cost, 2)

    # Inject a few missing values / minor data-quality issues realistically
    if random.random() < 0.02:
        cust_name = None
    if random.random() < 0.015:
        discount = None

    rows.append({
        "Order_ID": f"ORD-{order_num:05d}",
        "Order_Date": order_date.strftime("%Y-%m-%d"),
        "Customer_ID": cust_id,
        "Customer_Name": cust_name,
        "Customer_Segment": segment,
        "Product": product,
        "Category": category,
        "Region": region,
        "City": city,
        "Quantity": quantity,
        "Unit_Price": round(unit_price, 2),
        "Discount": discount,
        "Sales": sales,
        "Cost": cost,
        "Profit": profit,
    })

df = pd.DataFrame(rows)

# Introduce a handful of exact duplicate rows and a couple of outliers,
# matching the "realistic messy data" requirement.
dupes = df.sample(15, random_state=1)
df = pd.concat([df, dupes], ignore_index=True)
outlier_idx = df.sample(5, random_state=2).index
df.loc[outlier_idx, "Sales"] = df.loc[outlier_idx, "Sales"] * 12

output_path = Path(__file__).parent / "sample_sales_data.csv"
df.to_csv(output_path, index=False)
print(f"Generated {len(df):,} records -> {output_path}")
