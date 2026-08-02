#!/usr/bin/env python

import numpy as np
from control import WolframControl, ConstantControl
from tsetlin_fredkin import FredkinTsetlinMachine

# Parameters for the Tsetlin Machine
T = 15 
s = 3.9
number_of_clauses = 20
states = 100 

# Parameters of the pattern recognition problem
number_of_features = 12
number_of_classes = 2

# Training configuration
epochs = 200

# Loading of training and test data
training_data = np.loadtxt("NoisyXORTrainingData.txt").astype(dtype=np.int32)
test_data = np.loadtxt("NoisyXORTestData.txt").astype(dtype=np.int32)

X_training = training_data[:, 0:12].astype(np.int8)  # Input features
y_training = training_data[:, 12].astype(np.int32)   # Target value

X_test = test_data[:, 0:12].astype(np.int8)           # Input features
y_test = test_data[:, 12].astype(np.int32)            # Target value

print("--- Running Noisy XOR Demo with Fredkin CSWAP Routing ---")
tsetlin_machine = FredkinTsetlinMachine(
    number_of_classes=number_of_classes,
    number_of_clauses=number_of_clauses,
    number_of_features=number_of_features,
    number_of_states=states,
    s=s,
    threshold=T,
    seed=42,
    control_program=WolframControl(rule_number=30, boundary="wrap"),
    route_mode="features",
    edge_mode="xor",
    swap_when=1,
)

# Training of the Tsetlin Machine
tsetlin_machine.fit(X_training, y_training, epochs=epochs)

# Performance statistics
print(f"Accuracy on test data (no noise): {tsetlin_machine.evaluate(X_test, y_test):.4f}")
print(f"Accuracy on training data (40% noise): {tsetlin_machine.evaluate(X_training, y_training):.4f}")
print()
print("Prediction: x1 = 1, x2 = 0, ... -> y = ", tsetlin_machine.predict(np.array([1,0,1,1,1,0,1,1,1,0,0,0], dtype=np.int8)))
print("Prediction: x1 = 0, x2 = 1, ... -> y = ", tsetlin_machine.predict(np.array([0,1,1,1,1,0,1,1,1,0,0,0], dtype=np.int8)))
print("Prediction: x1 = 0, x2 = 0, ... -> y = ", tsetlin_machine.predict(np.array([0,0,1,1,1,0,1,1,1,0,0,0], dtype=np.int8)))
print("Prediction: x1 = 1, x2 = 1, ... -> y = ", tsetlin_machine.predict(np.array([1,1,1,1,1,0,1,1,1,0,0,0], dtype=np.int8)))
