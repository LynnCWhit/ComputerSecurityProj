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
