import pandas as pd
import matplotlib.pyplot as plt

# Load your cleaned dataset
df = pd.read_excel("Mobile_Shop_Management_Cleaned.xlsx")

# --- Analysis ---
# 1. Brands with highest stock (staying)
top_staying = df.groupby("Brand")["Stock Qty"].sum().sort_values(ascending=False).head(5)

# 2. Brands with lowest stock (leaving)
top_leaving = df.groupby("Brand")["Stock Qty"].sum().sort_values(ascending=True).head(5)

# 3. Price vs Stock correlation
price_stock_corr = df["Price (INR)"].corr(df["Stock Qty"])
print(f"Correlation between Price and Stock: {price_stock_corr:.2f}")

# --- Charts ---

# Bar Chart: Staying (High Stock)
plt.figure(figsize=(10,5))
top_staying.plot(kind="bar", color="skyblue", edgecolor="black")
plt.title("Top 5 Brands Staying (High Stock)")
plt.ylabel("Stock Quantity")
plt.xlabel("Brand")
plt.xticks(rotation=45)
plt.show()

# Bar Chart: Leaving (Low Stock)
plt.figure(figsize=(10,5))
top_leaving.plot(kind="bar", color="salmon", edgecolor="black")
plt.title("Top 5 Brands Leaving (Low Stock)")
plt.ylabel("Stock Quantity")
plt.xlabel("Brand")
plt.xticks(rotation=45)
plt.show()

# Scatter Plot: Price vs Stock
plt.figure(figsize=(8,6))
plt.scatter(df["Price (INR)"], df["Stock Qty"], alpha=0.7, c="purple")
plt.title("Price vs Stock Quantity (Are Expensive Phones Leaving Faster?)")
plt.xlabel("Price (INR)")
plt.ylabel("Stock Quantity")
plt.grid(True, linestyle="--", alpha=0.6)
plt.show()
