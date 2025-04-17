# CS4371
Anomaly detection simulation using Isolation Forest, LOF, and adaptive autoencoders.

# Clone Repository
git clone https://github.com/LynnCWhit/ComputerSecurityProj.git

# Build
pip or pip3 install -r requirements.txt

# Deploy
python3 or python "file_name.py"

# Functionality
Scatter plot generation based on different scenarios

    1. basic_detection_isoforest.py simulates a single run using an isoforest algorithm to detect what it believes to be fake interjected attack points in the data set without the cabability to learn.

    2. adaptive_detection_lof.py simulates 10 runs to create a line graph based off of detection rates for attacks contained in the data. Doesn't learn but is given positively detected data points from the previous run to determine detections in the current run

    3. adaptive_detection-autencoder.py simulates 10 runs to create a line graph based off of detection rates for attacks contained in the data. Has the capability to learn but it only learns once from data given from the first run to generalize attacks for consecutive runs 

    4. adaptive_autoencoder_cadstyle.py simulates 10 runs to create a line graph based off of detection rates for attacks contained in the data. Has the capability to learn and adapt based on every run that comes before the current run. 

    5. attack_resolution.py displays an example window that could be put into CAV vehicles to further improve the accuracy of attack detection rates but adding in human validation. Is not currently integrated into these simulations for prompting when an attack is assumed to be detected. 

# Reference
    Contemporary Reference (Follow-up / Forward-Looking):
Mammeri, A., Zrelli, M. H., & Zhou, J. (2023).
"Collaborative perception-based anomaly detection for autonomous driving systems."
Journal of Network and Computer Applications.
https://doi.org/10.1016/j.jnca.2023.103699

This paper discusses advanced anomaly detection systems in CAVs that use collaborative perception—just like the CAD system your project is trying to replicate and expand upon. It explores how sharing sensor data (like LiDAR) across vehicles increases performance but also the attack surface, reinforcing the need for adaptive and memory-informed anomaly detection models like your final autoencoder version.

How Your Code Relates to These Works:
basic_detection_isoforest.py: Reflects the foundational concept of detecting anomalies using static models. The model has no memory or adaptation—mirroring the early, simpler anomaly detection models.

adaptive_detection_lof.py: An intermediate step. It has limited adaptation (remembers the last run), using Local Outlier Factor, which bridges simple and learning-based models.

adaptive_detection_autoencoder.py (non-adaptive): Trains once—akin to a student cramming before exams. Reflects many current machine learning deployments that struggle with generalization over time.

adaptive_detection_autoencoder.py (adaptive): This version retrains after each run and best simulates the CAD system from your paper. It’s the most advanced and forward-looking, aligning with current trends in adversarial-aware and real-time anomaly detection in smart vehicles.

# What we learned


# How it Applies to what we have learned in class


# Application Areas


# Improvements
