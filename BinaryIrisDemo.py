#!/usr/bin/env python

import numpy as np
from control import WolframControl
from tsetlin_fredkin import FredkinTsetlinMachine

# Ensembles
ensemble_size = 5

# Parameters for the Tsetlin Machine
T = 10
s = 3.0
number_of_clauses = 300
states = 100 

# Parameters of the pattern recognition problem
number_of_features = 16
number_of_classes = 3

# Training configuration
epochs = 50

# Loading of training and test data
data = np.loadtxt("BinaryIrisData.txt").astype(dtype=np.int32)

accuracy_training = np.zeros(ensemble_size)
accuracy_test = np.zeros(ensemble_size)

for ensemble in range(ensemble_size):
    print("ENSEMBLE %d" % (ensemble + 1))
    print()

    np.random.shuffle(data)

    X_training = data[:int(data.shape[0]*0.8), 0:16].astype(np.int8)  # Input features
    y_training = data[:int(data.shape[0]*0.8), 16].astype(np.int32)  # Target value

    X_test = data[int(data.shape[0]*0.8):, 0:16].astype(np.int8)     # Input features
    y_test = data[int(data.shape[0]*0.8):, 16].astype(np.int32)      # Target value

    tsetlin_machine = FredkinTsetlinMachine(
        number_of_classes=number_of_classes,
        number_of_clauses=number_of_clauses,
        number_of_features=number_of_features,
        number_of_states=states,
        s=s,
        threshold=T,
        boost_true_positive_feedback=True,
        control_program=WolframControl(rule_number=110, boundary="wrap"),
        route_mode="features",
        edge_mode="xor",
        swap_when=1,
    )

    tsetlin_machine.fit(X_training, y_training, epochs=epochs)

    accuracy_test[ensemble] = tsetlin_machine.evaluate(X_test, y_test)
    accuracy_training[ensemble] = tsetlin_machine.evaluate(X_training, y_training)

    print("Average accuracy on test data: %.1f +/- %.1f" % (
        np.mean(100 * accuracy_test[:ensemble+1]),
        1.96 * np.std(100 * accuracy_test[:ensemble+1]) / np.sqrt(ensemble+1)
    ))
    print("Average accuracy on training data: %.1f +/- %.1f" % (
        np.mean(100 * accuracy_training[:ensemble+1]),
        1.96 * np.std(100 * accuracy_training[:ensemble+1]) / np.sqrt(ensemble+1)
    ))
    print()
