
import os
import joblib

# Load trained model
model = joblib.load("battery_model.pkl")

# Model file size
size_bytes = os.path.getsize("battery_model.pkl")
size_kb = size_bytes / 1024

# Extract neural network
mlp = model.named_steps["mlpregressor"]

# Count parameters
parameters = sum(w.size for w in mlp.coefs_)
parameters += sum(b.size for b in mlp.intercepts_)

print("MODEL INFORMATION")
print("-----------------")
print("Model file size:", round(size_kb, 2), "KB")
print("Trainable parameters:", parameters)
print("Architecture: 5 -> 32 -> 16 -> 1")
print("Activation:", mlp.activation)
print("Solver:", mlp.solver)
print("Iterations:", mlp.n_iter_)

# Estimate float32 weight storage
weight_memory = parameters * 4

print("\nESTIMATED FLOAT32 PARAMETER MEMORY")
print(weight_memory, "bytes")
print(round(weight_memory / 1024, 2), "KB")
