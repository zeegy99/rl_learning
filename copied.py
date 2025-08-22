# House Price Linear Regression - Fixed Version
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("csv_home_prices.csv")

# Normalize the data first
print("Original data ranges:")
print(f"Area: {df.area.min():.0f} - {df.area.max():.0f} sq ft")
print(f"Price: ${df.price.min():.0f} - ${df.price.max():.0f}")

# Store normalization parameters for later use
area_mean = df.area.mean()
area_std = df.area.std()
price_mean = df.price.mean()
price_std = df.price.std()

# Normalize features
df['area_norm'] = (df.area - area_mean) / area_std
df['price_norm'] = (df.price - price_mean) / price_std

print(f"\nNormalized data ranges:")
print(f"Area: {df.area_norm.min():.2f} - {df.area_norm.max():.2f}")
print(f"Price: {df.price_norm.min():.2f} - {df.price_norm.max():.2f}")

# Fixed gradient descent function
def gradient_descent(m_now, b_now, points, L):
    m_gradient = 0
    b_gradient = 0
    n = len(points)

    for i in range(n):
        x = points.iloc[i].area_norm
        y = points.iloc[i].price_norm

        # Calculate prediction error
        prediction = m_now * x + b_now
        error = y - prediction
        
        # Calculate gradients
        m_gradient += -(2/n) * x * error
        b_gradient += -(2/n) * error

    # Update parameters
    m = m_now - m_gradient * L
    b = b_now - b_gradient * L
    
    return (m, b)

# Training parameters
m = 0
b = 0
L = 0.01  # Much more reasonable learning rate with normalized data
epochs = 1000

# Training loop
for i in range(epochs):
    if i % 100 == 0:
        # Calculate cost for monitoring
        predictions = m * df.area_norm + b
        cost = np.mean((df.price_norm - predictions) ** 2)
        print(f"Epoch {i}: m={m:.4f}, b={b:.4f}, cost={cost:.4f}")
    
    m, b = gradient_descent(m, b, df, L)

print(f"\nFinal parameters (normalized): m={m:.4f}, b={b:.4f}")

# Convert back to original scale for interpretation
# price = m_norm * area_norm + b_norm
# price_std * price_norm + price_mean = m_norm * area_std * (area - area_mean)/area_std + b_norm
# price = m_norm * price_std/area_std * area + (b_norm * price_std + price_mean - m_norm * price_std * area_mean/area_std)

m_original = m * price_std / area_std
b_original = b * price_std + price_mean - m * price_std * area_mean / area_std

print(f"Final parameters (original scale): m={m_original:.2f}, b={b_original:.2f}")
print(f"Interpretation: Price = {m_original:.2f} * Area + {b_original:.2f}")
print(f"Meaning: Each additional sq ft increases price by ${m_original:.2f}")

# Plotting
plt.figure(figsize=(12, 5))

# Plot 1: Normalized data
plt.subplot(1, 2, 1)
plt.scatter(df.area_norm, df.price_norm, alpha=0.6)
x_range_norm = np.linspace(df.area_norm.min(), df.area_norm.max(), 100)
y_pred_norm = m * x_range_norm + b
plt.plot(x_range_norm, y_pred_norm, color='red', linewidth=2)
plt.xlabel('Normalized Area')
plt.ylabel('Normalized Price')
plt.title('Linear Regression (Normalized Data)')
plt.grid(True, alpha=0.3)

# Plot 2: Original scale
plt.subplot(1, 2, 2)
plt.scatter(df.area, df.price, alpha=0.6)
x_range_original = np.linspace(df.area.min(), df.area.max(), 100)
y_pred_original = m_original * x_range_original + b_original
plt.plot(x_range_original, y_pred_original, color='red', linewidth=2)
plt.xlabel('Area (sq ft)')
plt.ylabel('Price ($)')
plt.title('Linear Regression (Original Scale)')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Calculate R-squared
predictions_original = m_original * df.area + b_original
ss_res = np.sum((df.price - predictions_original) ** 2)
ss_tot = np.sum((df.price - df.price.mean()) ** 2)
r_squared = 1 - (ss_res / ss_tot)
print(f"\nModel Performance:")
print(f"R-squared: {r_squared:.4f}")
print(f"Mean Absolute Error: ${np.mean(np.abs(df.price - predictions_original)):.2f}")