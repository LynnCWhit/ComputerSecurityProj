import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

# Step 1: Generate 20 normal sensor points
num_points = 100
sensor_data = np.random.normal(loc=5, scale=1.2, size=(num_points, 2))

# Step 2: Simulate 5 fake points (attack)
num_fake_points = 20
fake_data = np.random.uniform(low=0, high=10, size=(num_fake_points, 2))

# Step 3: Combine the data
combined_data = np.vstack((sensor_data, fake_data))

# Step 4: Create ground truth labels (1 = real, -1 = fake)
ground_truth = np.array([1] * num_points + [-1] * num_fake_points)

# Step 5: Train Isolation Forest
model = IsolationForest(contamination=num_fake_points / (num_points + num_fake_points), random_state=42)
model.fit(combined_data)

# Step 6: Predict anomalies (-1 = anomaly, 1 = normal)
predictions = model.predict(combined_data)

# Step 7: Determine outcome categories for plotting
true_positives = combined_data[(ground_truth == -1) & (predictions == -1)]  # correctly flagged fakes
false_negatives = combined_data[(ground_truth == -1) & (predictions == 1)]  # missed fake points
true_negatives = combined_data[(ground_truth == 1) & (predictions == 1)]   # correctly identified normal
false_positives = combined_data[(ground_truth == 1) & (predictions == -1)]  # incorrectly flagged normal points

# Step 8: Plot all 3 stages side-by-side
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Step 9: Calculate performance metrics
total_attacks = num_fake_points
detected_attacks = len(true_positives)
missed_attacks = len(false_negatives)
false_alarms = len(false_positives)
total_normals = num_points

detection_rate = (detected_attacks / total_attacks) * 100
false_positive_rate = (false_alarms / total_normals) * 100
false_negative_rate = (missed_attacks / total_attacks) * 100

# Step 10: Print stats
print("----- Detection Performance -----")
print(f"Detection Rate: {detection_rate:.1f}% ({detected_attacks}/{total_attacks})")
print(f"False Positive Rate: {false_positive_rate:.1f}% ({false_alarms}/{total_normals})")
print(f"False Negative Rate: {false_negative_rate:.1f}% ({missed_attacks}/{total_attacks})")

# Plot 1: Original sensor data
axes[0].scatter(sensor_data[:, 0], sensor_data[:, 1], color='blue')
axes[0].set_title('Original Sensor Data')
axes[0].set_xlabel('X')
axes[0].set_ylabel('Y')
axes[0].grid(True)

# Plot 2: Data after attack
axes[1].scatter(sensor_data[:, 0], sensor_data[:, 1], color='blue', label='Normal')
axes[1].scatter(fake_data[:, 0], fake_data[:, 1], color='red', label='Fake (Attack)')
axes[1].set_title('With Attack (Fake Points Added)')
axes[1].set_xlabel('X')
axes[1].set_ylabel('Y')
axes[1].legend()
axes[1].grid(True)

# Plot 3: Detection Results with color-coded outcomes and stats
axes[2].scatter(true_negatives[:, 0], true_negatives[:, 1], color='green', label='True Normal')
axes[2].scatter(true_positives[:, 0], true_positives[:, 1], color='red', label='True Attack')
axes[2].scatter(false_positives[:, 0], false_positives[:, 1], color='yellow', label='False Alarm')
axes[2].scatter(false_negatives[:, 0], false_negatives[:, 1], color='gray', label='Missed Attack')

# Add performance stats to the title
axes[2].set_title(f'Detection Results\n'
                  f'Detection: {detection_rate:.1f}%, '
                  f'FP: {false_positive_rate:.1f}%, '
                  f'FN: {false_negative_rate:.1f}%')

axes[2].set_xlabel('X')
axes[2].set_ylabel('Y')
axes[2].legend()
axes[2].grid(True)

plt.tight_layout()
plt.show()