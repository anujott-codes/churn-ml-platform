import os

FOLDERS = [
    # Data layers
    "data/raw",
    "data/staged",
    "data/processed",

    # Source code
    "src",
    "src/config",

    "src/pipelines",
    "src/components",
    "src/utils",

    # Testing & infra
    "docker",
    ".github",
    ".github/workflows"
]

FILES = [
    # Pipelines
    "src/pipelines/training_pipeline.py",
    "src/pipelines/prediction_pipeline.py",

    # Utilities
    "src/utils/helpers.py",


    # Infra
    "docker/Dockerfile",
    ".github/workflows/ci.yml",

    # Root files
    "requirements.txt",
    "README.md"
]


def create_structure():
    # Create folders
    for folder in FOLDERS:
        os.makedirs(folder, exist_ok=True)

        # Add __init__.py to every Python package folder
        if folder.startswith("src") or folder.startswith("tests"):
            init_file = os.path.join(folder, "__init__.py")
            if not os.path.exists(init_file):
                open(init_file, "w").close()

    # Create files
    for file in FILES:
        if not os.path.exists(file):
            with open(file, "w"):
                pass

    print("Project structure created successfully...")


if __name__ == "__main__":
    create_structure()
