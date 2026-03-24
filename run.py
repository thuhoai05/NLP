# run.py

import subprocess

subprocess.run(["python", "build_index.py"])
subprocess.run(["python", "preprocess_for_training.py"])
subprocess.run(["python", "train.py"])
subprocess.run(["python", "app.py"])