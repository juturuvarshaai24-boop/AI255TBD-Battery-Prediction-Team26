
import time
import numpy as np
import onnxruntime as ort

sample = np.array(
    [[3.8, 28, 8, -65, 12]],
    dtype=np.float32
)

fp32 = ort.InferenceSession("battery_model.onnx")
int8 = ort.InferenceSession("battery_model_int8.onnx")

fp32_name = fp32.get_inputs()[0].name
int8_name = int8.get_inputs()[0].name

# Warm-up
for _ in range(200):
    fp32.run(None, {fp32_name: sample})
    int8.run(None, {int8_name: sample})


def benchmark(session, input_name):
    times = []

    for _ in range(2000):
        start = time.perf_counter()
        session.run(None, {input_name: sample})
        end = time.perf_counter()

        times.append((end - start) * 1000)

    return np.array(times)


fp32_times = benchmark(fp32, fp32_name)
int8_times = benchmark(int8, int8_name)

print("FP32 vs INT8 INFERENCE BENCHMARK")
print("================================")

for name, values in [
    ("ONNX FP32", fp32_times),
    ("ONNX INT8", int8_times)
]:
    print("\n", name)
    print("Average:", round(np.mean(values), 5), "ms")
    print("Median:", round(np.median(values), 5), "ms")
    print("P95:", round(np.percentile(values, 95), 5), "ms")

print("\nINT8 SPEEDUP:")
print(round(np.mean(fp32_times) / np.mean(int8_times), 3), "x")
