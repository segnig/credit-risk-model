import os

# Define folder and file structure
structure = {
    ".github/workflows": ["ci.yml"],
    "data/raw": [],
    "data/processed": [],
    "notebooks": ["1.0-eda.ipynb"],
    "src": ["__init__.py", "data_processing.py", "train.py", "predict.py"],
    "src/api": ["main.py", "pydantic_models.py"],
    "tests": ["test_data_processing.py"],
    ".": ["Dockerfile", "docker-compose.yml", "requirements.txt", ".gitignore", "README.md"]
}

# Create folders and files
for folder, files in structure.items():
    os.makedirs(folder, exist_ok=True)
    for file in files:
        file_path = os.path.join(folder, file)
        if not os.path.exists(file_path):
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(f"# {file}\n")
            print(f"Created: {file_path}")
        else:
            print(f"Already exists: {file_path}")