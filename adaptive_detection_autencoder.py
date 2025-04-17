import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# ----- Autoencoder Definition -----
class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(2, 4),
            nn.ReLU(),
            nn.Linear(4, 2)
        )
        self.decoder = nn.Sequential(
            nn.Linear(2, 4),
            nn.ReLU(),
            nn.Linear(4, 2),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# ----- Step 1: Train the Autoencoder -----
num_train_points = 1000
train_data = np.random.normal(loc=5, scale=1.2, size=(num_train_points, 2))

scaler = MinMaxScaler()
train_scaled = scaler.fit_transform(train_data)
train_tensor = torch.tensor(train_scaled, dtype=torch.float32)

model = Autoencoder()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)
epochs = 50

for epoch in range(epochs):
    output = model(train_tensor)
    loss = criterion(output, train_tensor)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.6f}")

# ----- Step 2: Multi-Run Evaluation -----
num_runs = 10
detection_rates = []
false_positive_rates = []
false_negative_rates = []

for run in range(num_runs):
    # Generate fresh normal + fake test data
    test_normal = np.random.normal(loc=5, scale=1.2, size=(100, 2))
    test_fake = np.random.uniform(low=0, high=10, size=(20, 2))
    test_combined = np.vstack((test_normal, test_fake))
    test_scaled = scaler.transform(test_combined)
    test_tensor = torch.tensor(test_scaled, dtype=torch.float32)

    # Run through autoencoder
    model.eval()
    with torch.no_grad():
        reconstructed = model(test_tensor)

    reconstruction_error = torch.mean((test_tensor - reconstructed) ** 2, dim=1).numpy()
    threshold = np.percentile(reconstruction_error, 80)
    predicted_anomalies = reconstruction_error > threshold

    # Accuracy metrics
    true_labels = np.array([0] * 100 + [1] * 20)  # 0 = normal, 1 = fake
    predicted_labels = predicted_anomalies.astype(int)

    tp = np.sum((true_labels == 1) & (predicted_labels == 1))
    fp = np.sum((true_labels == 0) & (predicted_labels == 1))
    fn = np.sum((true_labels == 1) & (predicted_labels == 0))

    detection_rate = (tp / 20) * 100
    false_positive_rate = (fp / 100) * 100
    false_negative_rate = (fn / 20) * 100

    detection_rates.append(detection_rate)
    false_positive_rates.append(false_positive_rate)
    false_negative_rates.append(false_negative_rate)

    print(f"\n--- Run {run + 1} ---")
    print(f"Detection Rate: {detection_rate:.1f}%")
    print(f"False Positive Rate: {false_positive_rate:.1f}%")
    print(f"False Negative Rate: {false_negative_rate:.1f}%")

# ----- Step 3: Plot performance over time -----
plt.figure(figsize=(10, 6))
runs = range(1, num_runs + 1)

plt.plot(runs, detection_rates, label="Detection Rate", marker='o')
plt.plot(runs, false_positive_rates, label="False Positive Rate", marker='x')
plt.plot(runs, false_negative_rates, label="False Negative Rate", marker='s')

# Annotate each point
for i, (d, fp, fn) in enumerate(zip(detection_rates, false_positive_rates, false_negative_rates)):
    plt.text(runs[i], d + 1, f"{d:.1f}%", color='green', fontsize=8, ha='center')
    plt.text(runs[i], fp + 1, f"{fp:.1f}%", color='red', fontsize=8, ha='center')
    plt.text(runs[i], fn + 1, f"{fn:.1f}%", color='gray', fontsize=8, ha='center')

plt.title("Autoencoder Anomaly Detection Over Multiple Runs")
plt.xlabel("Run Number")
plt.ylabel("Percentage")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
