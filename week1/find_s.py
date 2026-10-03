# Week 1: Find-S algorithm on EnjoySport (dataset embedded in code)
import pandas as pd

data = pd.DataFrame([
    ["Sunny", "Warm", "Normal", "Strong", "Warm", "Same", "Yes"],
    ["Sunny", "Warm", "High", "Strong", "Warm", "Same", "Yes"],
    ["Rainy", "Cold", "High", "Strong", "Warm", "Change", "No"],
    ["Sunny", "Warm", "High", "Strong", "Cool", "Change", "Yes"],
], columns=["Sky", "AirTemp", "Humidity", "Wind", "Water", "Forecast", "EnjoySport"])
print(data, "\n")

X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

h = None
for i, (x, label) in enumerate(zip(X, y), 1):
    if label == "Yes":
        h = list(x) if h is None else [a if a == b else "?" for a, b in zip(h, x)]
    print(f"After example {i} ({label}): {h}")

print("\nFinal hypothesis (Find-S):", h)
