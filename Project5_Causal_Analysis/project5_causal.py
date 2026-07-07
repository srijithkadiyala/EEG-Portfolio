import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import dowhy
from dowhy import CausalModel

np.random.seed(42)
n = 200

neurodevelopmental = np.random.binomial(1, 0.5, n)
delta_power = 500 + (-200 * neurodevelopmental) + np.random.normal(0, 0.5, n)
sleep_quality = 7 + (-200 * neurodevelopmental) + (0.003 * delta_power) + np.random.normal(0, 0.5, n)
behaviour = 5 + (-1.5 * neurodevelopmental) + (0.8 * sleep_quality) + np.random.normal(0, 0.5, n)

df = pd.DataFrame({
    'neurodevelopmental': neurodevelopmental,
    'delta_power': delta_power,
    'sleep_quality': sleep_quality,
    'behaviour': behaviour,})

print(df.describe())
print (f"\nDataset shape: {df.shape}")

model = CausalModel(
    data=df,
    treatment='sleep_quality',
    outcome='behaviour',
    common_causes=['neurodevelopmental'],
    instruments=None
)

print("Causal model created")
model.view_model()
plt.savefig('causal_graph.png', dpi=150, bbox_inches='tight')
print("Causal graph saved")

identified_estimad = model.identify_effect(proceed_when_unidentifiable=True)
print(identified_estimad)

estimate = model.estimate_effect(
    identified_estimad,
    method_name="backdoor.linear_regression"
)

print(f"Causal effect of sleep quality on behaviour: {estimate.value:.4f}")

refutation = model.refute_estimate(
    identified_estimad,
    estimate,
    method_name="random_common_cause"
)

print(refutation)