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

# ----- Initial Training on Normal Data -----
num_initial = 1000
initial_data = np.random.normal(loc=5, scale=1.2, size=(num_initial, 2))
scaler = MinMaxScaler()
scaled_initial = scaler.fit_transform(initial_data)
tensor_initial = torch.tensor(scaled_initial, dtype=torch.float32)

model = Autoencoder()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Train once on the initial normal data
epochs = 50
for epoch in range(epochs):
    output = model(tensor_initial)
    loss = criterion(output, tensor_initial)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# ----- Adaptive Loop -----
num_runs = 10
detection_rates = []
false_positive_rates = []
false_negative_rates = []

for run in range(num_runs):
    # --- Generate test data ---
    normal_test = np.random.normal(loc=5, scale=1.2, size=(100, 2))
    fake_test = np.random.uniform(low=0, high=10, size=(20, 2))
    combined_test = np.vstack((normal_test, fake_test))
    scaled_test = scaler.transform(combined_test)
    test_tensor = torch.tensor(scaled_test, dtype=torch.float32)

    # --- Run inference ---
    model.eval()
    with torch.no_grad():
        recon = model(test_tensor)

    reconstruction_error = torch.mean((test_tensor - recon) ** 2, dim=1).numpy()
    threshold = np.percentile(reconstruction_error, 80)
    predicted_anomalies = reconstruction_error > threshold
    predicted_labels = predicted_anomalies.astype(int)

    # Ground truth: 0 = normal, 1 = fake
    true_labels = np.array([0] * 100 + [1] * 20)

    # Metrics
    tp = np.sum((true_labels == 1) & (predicted_labels == 1))
    fp = np.sum((true_labels == 0) & (predicted_labels == 1))
    fn = np.sum((true_labels == 1) & (predicted_labels == 0))

    detection_rate = (tp / 20) * 100
    false_pos_rate = (fp / 100) * 100
    false_neg_rate = (fn / 20) * 100

    detection_rates.append(detection_rate)
    false_positive_rates.append(false_pos_rate)
    false_negative_rates.append(false_neg_rate)

    # --- Extract new trusted normal points (predicted normal & known normal) ---
    trusted_normals = scaled_test[(predicted_labels == 0) & (true_labels == 0)]
    if len(trusted_normals) > 0:
        trusted_tensor = torch.tensor(trusted_normals, dtype=torch.float32)

        # --- Update model on trusted normals ---
        model.train()
        for _ in range(10):  # small retrain
            out = model(trusted_tensor)
            loss = criterion(out, trusted_tensor)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # --- Print run summary ---
    print(f"\n--- Run {run + 1} ---")
    print(f"Detection Rate: {detection_rate:.1f}%")
    print(f"False Positive Rate: {false_pos_rate:.1f}%")
    print(f"False Negative Rate: {false_neg_rate:.1f}%")

# ----- Final Graph -----
plt.figure(figsize=(10, 6))
runs = range(1, num_runs + 1)
plt.plot(runs, detection_rates, label="Detection Rate", marker='o')
plt.plot(runs, false_positive_rates, label="False Positive Rate", marker='x')
plt.plot(runs, false_negative_rates, label="False Negative Rate", marker='s')

for i, (d, fp, fn) in enumerate(zip(detection_rates, false_positive_rates, false_negative_rates)):
    plt.text(runs[i], d + 1, f"{d:.1f}%", color='green', fontsize=8, ha='center')
    plt.text(runs[i], fp + 1, f"{fp:.1f}%", color='red', fontsize=8, ha='center')
    plt.text(runs[i], fn + 1, f"{fn:.1f}%", color='gray', fontsize=8, ha='center')

plt.title("Adaptive Autoencoder Anomaly Detection Over Time")
plt.xlabel("Run Number")
plt.ylabel("Percentage")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
