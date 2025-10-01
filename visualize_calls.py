import pandas as pd
import matplotlib.pyplot as plt

# Load the synthetic dataset
df = pd.read_csv("synthetic_calls.csv")

# Color map for incident types
colors = {
    "Traffic Accident": "red",
    "Fire": "orange",
    "Medical Emergency": "green",
    "Other": "blue"
}

# Create the plot
plt.figure(figsize=(8, 8))
for incident, group in df.groupby("IncidentType"):
    plt.scatter(group["X"], group["Y"], 
                label=incident, 
                c=colors[incident], 
                alpha=0.6, 
                edgecolor="k")

# Add details
plt.title("Emergency Incident Locations in Synthetic City Grid")
plt.xlabel("X Coordinate")
plt.ylabel("Y Coordinate")
plt.xlim(0, 100)
plt.ylim(0, 100)
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)

# Show plot
plt.show()
