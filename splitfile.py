import shutil
import os
import random
from pathlib import Path

SOURCE_DIR = "/home/nithish/quantum_circuit_gnn/training_data"

TRAIN_DIR = "/home/nithish/quantum_circuit_gnn_plots_3/train"
TEST_DIR = "/home/nithish/quantum_circuit_gnn_plots_3/test"

TEST_RATIO = 0.1
SEED = 42

graph_files = []
for ext in ("*.gpickle", "*.pkl", "*.pickle"):
    graph_files.extend(Path(SOURCE_DIR).rglob(ext))

graph_files = [str(f) for f in graph_files]

print(f"Found {len(graph_files)} graph files")

if len(graph_files) == 0:
    raise RuntimeError(
        f"No graph files found inside {SOURCE_DIR}"
    )

random.seed(SEED)
random.shuffle(graph_files)

n_total = len(graph_files)
n_test = int(n_total * TEST_RATIO)

test_files = graph_files[:n_test]
train_files = graph_files[n_test:]

print(f"Train files: {len(train_files)}")
print(f"Test files : {len(test_files)}")

os.makedirs(TRAIN_DIR, exist_ok=True)
os.makedirs(TEST_DIR, exist_ok=True)

for f in train_files:
    shutil.copy2(f, os.path.join(TRAIN_DIR, os.path.basename(f)))

for f in test_files:
    shutil.copy2(f, os.path.join(TEST_DIR, os.path.basename(f)))

print("\nFinished")
print(f"Train folder : {TRAIN_DIR}")
print(f"Test folder  : {TEST_DIR}")

