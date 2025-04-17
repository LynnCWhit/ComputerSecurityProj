import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import LocalOutlierFactor

# Simulation parameters
num_runs = 10
num_points = 100
num_fake_points = 20
trusted_memory = []

# For tracking accuracy over time
detection_rates = []
false_positive_rates = []
false_negative_rates = []

for run in range(num_runs):
    # --- Step 1: Generate normal + fake data
    new_normal = np.random.normal(loc=5, scale=1.2, size=(num_points, 2))
    fake_data = np.random.uniform(low=0, high=10, size=(num_fake_points, 2))

    # --- Step 2: Flatten and limit trusted memory
    if trusted_memory:
        flattened_memory = np.vstack(trusted_memory)
        if len(flattened_memory) > 200:
            flattened_memory = flattened_memory[-200:]  # keep last 200 points
        trusted_memory = [flattened_memory]
        combined_data = np.vstack((new_normal, fake_data, flattened_memory))
        trusted_memory_len = len(flattened_memory)
    else:
        combined_data = np.vstack((new_normal, fake_data))
        trusted_memory_len = 0

    # --- Step 3: Ground truth (1 = real, -1 = fake)
    ground_truth = np.array(
        [1] * num_points + [-1] * num_fake_points + [1] * trusted_memory_len
    )

    # --- Step 4: LOF detection with higher neighbors
    lof = LocalOutlierFactor(
        n_neighbors=40,
        contamination=num_fake_points / (num_points + num_fake_points),
    )
    predictions = lof.fit_predict(combined_data)

    # --- Step 5: Categorize predictions
    true_positives = combined_data[(ground_truth == -1) & (predictions == -1)]
    false_negatives = combined_data[(ground_truth == -1) & (predictions == 1)]
    true_negatives = combined_data[(ground_truth == 1) & (predictions == 1)]
    false_positives = combined_data[(ground_truth == 1) & (predictions == -1)]

    # --- Step 6: Accuracy tracking
    detection_rate = (len(true_positives) / num_fake_points) * 100
    false_pos_rate = (len(false_positives) / (num_points + trusted_memory_len)) * 100
    false_neg_rate = (len(false_negatives) / num_fake_points) * 100

    detection_rates.append(detection_rate)
    false_positive_rates.append(false_pos_rate)
    false_negative_rates.append(false_neg_rate)

    # --- Step 7: Update trusted memory with confident normals
    trusted_normals = combined_data[(ground_truth == 1) & (predictions == 1)]
    trusted_memory.append(trusted_normals)

    # --- Step 8: Output summary
    print(f"\n--- Run {run + 1} ---")
    print(f"Detection Rate: {detection_rate:.1f}%")
    print(f"False Positive Rate: {false_pos_rate:.1f}%")
    print(f"False Negative Rate: {false_neg_rate:.1f}%")

# --- Final plot: performance over time
plt.figure(figsize=(10, 6))
runs = range(1, num_runs + 1)

plt.plot(runs, detection_rates, label="Detection Rate", marker='o')
plt.plot(runs, false_positive_rates, label="False Positive Rate", marker='x')
plt.plot(runs, false_negative_rates, label="False Negative Rate", marker='s')

# Annotate each point with percentages
for i, (d, fp, fn) in enumerate(zip(detection_rates, false_positive_rates, false_negative_rates)):
    plt.text(runs[i], d + 1, f"{d:.1f}%", color='green', fontsize=8, ha='center')
    plt.text(runs[i], fp + 1, f"{fp:.1f}%", color='red', fontsize=8, ha='center')
    plt.text(runs[i], fn + 1, f"{fn:.1f}%", color='gray', fontsize=8, ha='center')

plt.title("Performance Over Time with LOF + Memory")
plt.xlabel("Run Number")
plt.ylabel("Percentage")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
