import subprocess

print("Step 1: Build index")
subprocess.run(["python", "build_index.py"])

print("Step 2: Train model")
subprocess.run(["python", "train.py"])

print("Step 3: Run chatbot")
subprocess.run(["python", "chatbot.py"])