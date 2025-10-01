import simpy
import random
import numpy as np
import matplotlib.pyplot as plt
import math

# Parameters
RANDOM_SEED = 42
NUM_CALLS = 1000
NUM_AMBULANCES = 5
CITY_SIZE = 100
AMBULANCE_SPEED = 1.5  # units per minute

SEVERITY_DIST = {"critical": 0.2, "moderate": 0.5, "low": 0.3}
SEVERITY_SERVICE = {"critical": 10, "moderate": 15, "low": 20}

# Ambulance bases (hospitals)
HOSPITALS = [(10, 10), (90, 10), (50, 50), (10, 90), (90, 90)]

# Stats
wait_times = {"critical": [], "moderate": [], "low": []}
response_times = {"critical": [], "moderate": [], "low": []}
travel_times = []

def distance(p1, p2):
    return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

class Ambulance:
    def __init__(self, id, base):
        self.id = id
        self.location = base  # current location
        self.status = "Idle"

    def move_to(self, new_location):
        self.location = new_location

def emergency_call(env, name, ambulances, severity, location):
    arrival_time = env.now
    priority = {"critical": 0, "moderate": 1, "low": 2}[severity]

    with ambulances.request(priority=priority) as req:
        yield req
        wait = env.now - arrival_time
        wait_times[severity].append(wait)

        # Pick nearest ambulance dynamically
        amb = min(ambulance_list, key=lambda a: distance(a.location, location))

        dist = distance(amb.location, location)
        travel_time = dist / AMBULANCE_SPEED
        travel_times.append(travel_time)

        amb.status = "Dispatched"
        yield env.timeout(travel_time)

        # Service
        service_time = random.expovariate(1.0 / SEVERITY_SERVICE[severity])
        yield env.timeout(service_time)

        amb.status = "Transporting"
        # Drop at nearest hospital
        nearest_hospital = min(HOSPITALS, key=lambda h: distance(location, h))
        dist_back = distance(location, nearest_hospital)
        yield env.timeout(dist_back / AMBULANCE_SPEED)

        amb.move_to(nearest_hospital)
        amb.status = "Idle"

        response_times[severity].append(env.now - arrival_time)

def call_generator(env, ambulances):
    for i in range(NUM_CALLS):
        inter_arrival = random.expovariate(1.0 / 5.0)
        yield env.timeout(inter_arrival)

        severity = random.choices(
            population=list(SEVERITY_DIST.keys()),
            weights=list(SEVERITY_DIST.values())
        )[0]

        # Random incident location
        x, y = random.uniform(0, CITY_SIZE), random.uniform(0, CITY_SIZE)
        env.process(emergency_call(env, f"Call {i+1}", ambulances, severity, (x, y)))

# Run simulation
random.seed(RANDOM_SEED)
env = simpy.Environment()
ambulances = simpy.PriorityResource(env, NUM_AMBULANCES)

# Track individual ambulances
ambulance_list = [Ambulance(i, base) for i, base in enumerate(HOSPITALS)]

env.process(call_generator(env, ambulances))
env.run()

# Results
for sev in wait_times:
    print(f"\n{sev.upper()} CALLS:")
    print(f"  Avg wait time: {np.mean(wait_times[sev]):.2f}")
    print(f"  Avg response time: {np.mean(response_times[sev]):.2f}")

print(f"\nAvg travel time per dispatch: {np.mean(travel_times):.2f} minutes")

# Visualization of final ambulance positions
plt.figure(figsize=(7,7))
for h in HOSPITALS:
    plt.scatter(*h, c="blue", marker="s", s=100, label="Hospital" if h==HOSPITALS[0] else "")

for amb in ambulance_list:
    plt.scatter(*amb.location, c="green", marker="^", s=100, label="Ambulance" if amb.id==0 else "")

xs = [random.uniform(0,CITY_SIZE) for _ in range(50)]
ys = [random.uniform(0,CITY_SIZE) for _ in range(50)]
plt.scatter(xs, ys, c="red", alpha=0.5, label="Calls")

plt.title("City grid with hospitals, ambulances, and incident calls")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.grid(True)
plt.show()
