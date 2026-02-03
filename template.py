import os

FOLDERS = [
    # Data layers
    "data/raw",
    "data/staged",
    "data/processed",

    # Source code
    "src",
    "src/config",

    "src/etl",
    "src/etl/extract",
    "src/etl/validate",
    "src/etl/transform",
    "src/etl/load",

    "src/pipelines",
    "src/components",
    "src/mlops",
    "src/utils",

    # Application
    "src/app",
    "src/app/backend",
    "src/app/frontend",

    # Testing & infra
    "tests",
    "docker",
    ".github",
    ".github/workflows"
]

FILES = [
    # Config
    "src/config/database.py",
    "src/config/settings.py",

    # ETL components
    "src/etl/extract/extract_raw.py",
    "src/etl/validate/validate_data.py",
    "src/etl/transform/feature_engineering.py",
    "src/etl/load/load_raw.py",
    "src/etl/load/load_features.py",

    # Pipelines
    "src/pipelines/etl_pipeline.py",
    "src/pipelines/training_pipeline.py",
    "src/pipelines/prediction_pipeline.py",

    # Model logic
    "src/components/model_trainer.py",
    "src/components/model_evaluator.py",
    "src/components/model_predictor.py",

    # MLOps
    "src/mlops/model_registry.py",
    "src/mlops/tracking.py",

    # Utilities
    "src/utils/logger.py",
    "src/utils/helpers.py",

    # Web app
    "src/app/backend/api.py",
    "src/app/frontend/dashboard.py",

    # Tests
    "tests/test_etl.py",

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
