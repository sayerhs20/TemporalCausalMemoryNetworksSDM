import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)

COUNTRIES = ["India", "USA", "UK", "Singapore", "Canada"]
COUNTRY_CANCEL_BIAS = {"India": -0.10, "USA": 0.05, "UK": 0.00, "Singapore": -0.05, "Canada": 0.10}
CATEGORIES = ["Electronics", "Accessories", "Office", "Storage", "Networking"]


def generate_synthetic_orders(n=250, start_id=500):
    rows = []
    for i in range(n):
        order_id = start_id + i
        customer_id = 300 + RNG.integers(1, 400)
        country = RNG.choice(COUNTRIES)
        category = RNG.choice(CATEGORIES)

        price1 = round(RNG.uniform(400, 5000), 2)
        step1_change = RNG.normal(0, 0.05)          
        price2 = round(price1 * (1 + step1_change), 2)
        step2_change = RNG.normal(0.02, 0.08)        
        price3 = round(price2 * (1 + step2_change), 2)

        late_increase = (price3 - price2) / price2
        cancel_logit = 2.5 * late_increase + COUNTRY_CANCEL_BIAS[country]
        cancel_prob = 1 / (1 + np.exp(-cancel_logit * 5))

        roll = RNG.random()
        if roll < cancel_prob * 0.6:
            status = "Cancelled"
        elif roll < cancel_prob * 0.6 + 0.15:
            status = "Pending"
        else:
            status = "Completed"

        purchase_date = pd.Timestamp("2025-01-01") + pd.Timedelta(days=int(RNG.integers(0, 240)))

        rows.append({
            "order_id": order_id,
            "customer_id": customer_id,
            "status": status,
            "price1": price1,
            "price2": price2,
            "price3": price3,
            "purchase_date": purchase_date.strftime("%Y-%m-%d"),
            "country": country,
            "category": category,
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    real_orders = pd.read_csv("data/orders.csv")
    customers = pd.read_csv("data/customers.csv")
    real_orders = real_orders.merge(customers[["customer_id", "country"]], on="customer_id", how="left")
    real_orders["category"] = "Electronics"

    synthetic = generate_synthetic_orders(n=250)
    combined = pd.concat([real_orders, synthetic], ignore_index=True)
    combined.to_csv("data/orders_expanded.csv", index=False)
    print(f"Wrote {len(combined)} orders to data/orders_expanded.csv")
    print(combined["status"].value_counts())
