import os
from pathlib import Path

project_name = "src"

list_of_files = [

    f"mlops-project-/{project_name}/__init__.py",
    f"mlops-project-/{project_name}/components/__init__.py",
    f"mlops-project-/{project_name}/components/data_ingestion.py",  
    f"mlops-project-/{project_name}/components/data_validation.py",
    f"mlops-project-/{project_name}/components/data_transformation.py",
    f"mlops-project-/{project_name}/components/model_trainer.py",
    f"mlops-project-/{project_name}/components/model_evaluation.py",
    f"mlops-project-/{project_name}/components/model_pusher.py",
    f"mlops-project-/{project_name}/configuration/__init__.py",
    f"mlops-project-/{project_name}/configuration/mongo_db_connection.py",
    f"mlops-project-/{project_name}/configuration/aws_connection.py",
    f"mlops-project-/{project_name}/cloud_storage/__init__.py",
    f"mlops-project-/{project_name}/cloud_storage/aws_storage.py",
    f"mlops-project-/{project_name}/data_access/__init__.py",
    f"mlops-project-/{project_name}/data_access/proj1_data.py",
    f"mlops-project-/{project_name}/constants/__init__.py",
    f"mlops-project-/{project_name}/entity/__init__.py",
    f"mlops-project-/{project_name}/entity/config_entity.py",
    f"mlops-project-/{project_name}/entity/artifact_entity.py",
    f"mlops-project-/{project_name}/entity/estimator.py",
    f"mlops-project-/{project_name}/entity/s3_estimator.py",
    f"mlops-project-/{project_name}/exception/__init__.py",
    f"mlops-project-/{project_name}/logger/__init__.py",
    f"mlops-project-/{project_name}/pipline/__init__.py",
    f"mlops-project-/{project_name}/pipline/training_pipeline.py",
    f"mlops-project-/{project_name}/pipline/prediction_pipeline.py",
    f"mlops-project-/{project_name}/utils/__init__.py",
    f"mlops-project-/{project_name}/utils/main_utils.py",
    "mlops-project-/app.py",
    "mlops-project-/requirements.txt",
    "mlops-project-/Dockerfile",
    "mlops-project-/.dockerignore",
    "mlops-project-/demo.py",
    "mlops-project-/setup.py",
    "mlops-project-/pyproject.toml",
    "mlops-project-/config/model.yaml",
    "mlops-project-/config/schema.yaml",
]


for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)
    if filedir != "":
        os.makedirs(filedir, exist_ok=True)
    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath, "w") as f:
            pass
    else:
        print(f"file is already present at: {filepath}")