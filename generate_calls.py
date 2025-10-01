import pandas as pd
import numpy as np
from faker import Faker

# initialize Faker and random seed for reproducibility
fake = Faker()
np.random.seed(42)

# parameters
NUM_CALLS = 1000
CITY_GRID_SIZE = 100  # 100x100 virtual city

# probability distributions
incident_types = ["Traffic Accident", "Fire", "Medical Emergency", "Other"]
incident_probs = [0.4, 0.2, 0.3, 0.1]  # weighted likelihood

severity_levels = ["Critical", "Moderate", "Low"]
severity_probs = [0.2, 0.5, 0.3]  # weighted likelihood

response_units = ["Ambulance", "Fire Truck", "Police", "Mixed"]
response_probs = [0.5, 0.2, 0.2, 0.1]

# Generate synthetic data
data = []
for i in range(NUM_CALLS):
    call_id = i + 1
    timestamp = fake.date_time_this_year()  # random time this year
    x_coord = np.random.randint(0, CITY_GRID_SIZE)
    y_coord = np.random.randint(0, CITY_GRID_SIZE)
    incident = np.random.choice(incident_types, p=incident_probs)
    severity = np.random.choice(severity_levels, p=severity_probs)
    response = np.random.choice(response_units, p=response_probs)
    
    data.append([call_id, timestamp, x_coord, y_coord, incident, severity, response])

# Create DataFrame
df = pd.DataFrame(data, columns=["CallID", "Timestamp", "X", "Y", "IncidentType", "Severity", "ResponseUnit"])

# Save to CSV
df.to_csv("synthetic_calls.csv", index=False)

print("✅ Synthetic dataset generated: synthetic_calls.csv")
print(df.head(10))
