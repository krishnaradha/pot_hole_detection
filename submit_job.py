from azure.ai.ml import MLClient, command, Input
from azure.ai.ml.entities import Environment
from azure.identity import DefaultAzureCredential
from dotenv import load_dotenv
from pathlib import Path
import os

# ----------------------------------------------------------------------------- #
# Load environment variables
# ----------------------------------------------------------------------------- #
load_dotenv()

subscription_id = os.getenv("AZURE_SUBSCRIPTION_ID")
resource_group = os.getenv("AZURE_RESOURCE_GROUP")
workspace_name = os.getenv("AZURE_WORKSPACE_NAME")
compute_cluster = os.getenv("COMPUTE_NAME")
datastore_name = os.getenv("DATASTORE_NAME")  # 👈 points to blob datastore

if not all([subscription_id, resource_group, workspace_name, compute_cluster, datastore_name]):
    raise ValueError("❌ One or more required environment variables are missing in .env")

# ----------------------------------------------------------------------------- #
# Initialize ML client
# ----------------------------------------------------------------------------- #
ml_client = MLClient(
    DefaultAzureCredential(),
    subscription_id=subscription_id,
    resource_group_name=resource_group,
    workspace_name=workspace_name
)

# ----------------------------------------------------------------------------- #
# Get existing environment
# ----------------------------------------------------------------------------- #
env_name = "od_env"
env_path = Path(__file__).parent / "environment.yml"
try:
    env = ml_client.environments.get(name=env_name, label="latest")
    print(f"✅ Environment '{env_name}' found.")
except:
    env = Environment(
        name=env_name,
        conda_file=env_path,
        image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04"
    )
    ml_client.environments.create_or_update(env)
    print(f"✅ Environment '{env_name}' created.")

# ----------------------------------------------------------------------------- #
# Define datastore input (auto-mounted during training)
# ----------------------------------------------------------------------------- #
data_input = Input(
    type="uri_folder",
    path=f"azureml://datastores/{datastore_name}/paths/",  # adjust prefix
    mode="mount"
)

# ----------------------------------------------------------------------------- #
# Submit the YOLO training job
# ----------------------------------------------------------------------------- #
job = command(
    code=str(Path(__file__).parent),
    command="python train.py",
    inputs={"data": data_input},  # makes /mnt/data available during run
    environment=env,
    compute=compute_cluster,
    display_name="pothole-yolo-training",
    experiment_name="pothole_detection_experiment",
    description="YOLOv8 training job for pothole detection",
    outputs={
        "model_output": {"mode": "rw_mount", "type": "uri_folder"}  # captures outputs/
    }
)

# ----------------------------------------------------------------------------- #
# Submit to Azure ML
# ----------------------------------------------------------------------------- #
returned_job = ml_client.jobs.create_or_update(job)
print(f"🚀 Job submitted successfully! Job name: {returned_job.name}")
print(f"🔗 View in Azure ML Studio: {returned_job.studio_url}")
