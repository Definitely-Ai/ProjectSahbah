py -3.12 -m venv .venv312
.\.venv312\Scripts\python.exe -m pip install --upgrade pip
.\.venv312\Scripts\python.exe -m pip install -e ".[dev,live]"
.\.venv312\Scripts\python.exe pose_jitter.py cameras
