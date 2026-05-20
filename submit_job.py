import os
from pathlib import Path

from azure.ai.ml import MLClient, Input, command
from azure.ai.ml.entities import Environment
from azure.identity import DefaultAzureCredential
from dotenv import load_dotenv


load_dotenv()

subscription_id = os.getenv("AZURE_SUBSCRIPTION_ID")
resource_group = os.getenv("AZURE_RESOURCE_GROUP")
workspace_name = os.getenv("AZURE_WORKSPACE_NAME")
compute_cluster = os.getenv("COMPUTE_NAME")
datastore_name = os.getenv("DATASTORE_NAME")

if not all([subscription_id, resource_group, workspace_name, compute_cluster, datastore_name]):
    raise ValueError("One or more required environment variables are missing in .env")

ml_client = MLClient(
    DefaultAzureCredential(),
    subscription_id=subscription_id,
    resource_group_name=resource_group,
    workspace_name=workspace_name,
)

env_name = "od_env"
env_path = Path(__file__).parent / "environment.yml"
try:
    env = ml_client.environments.get(name=env_name, label="latest")
    print(f"Environment '{env_name}' found.")
except Exception:
    env = Environment(
        name=env_name,
        conda_file=env_path,
        image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04",
    )
    ml_client.environments.create_or_update(env)
    print(f"Environment '{env_name}' created.")

data_input = Input(
    type="uri_folder",
    path=f"azureml://datastores/{datastore_name}/paths/",
    mode="mount",
)

job = command(
    code=str(Path(__file__).parent),
    command="python train.py",
    inputs={"data": data_input},
    environment=env,
    compute=compute_cluster,
    display_name="pothole-yolo-training",
    experiment_name="pothole_detection_experiment",
    description="YOLOv8 training job for pothole detection",
    outputs={
        "model_output": {"mode": "rw_mount", "type": "uri_folder"},
    },
)

returned_job = ml_client.jobs.create_or_update(job)
print(f"Job submitted: {returned_job.name}")
print(f"View in Azure ML Studio: {returned_job.studio_url}")
