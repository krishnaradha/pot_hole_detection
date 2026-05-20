import os
from azure.storage.blob import BlobServiceClient
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
conn_str = os.getenv("AZURE_CONN_STR")
container_name = os.getenv("AZURE_CONTAINER_NAME")

# Separate local paths
images_path = r"C:\Users\bhagy\OneDrive - sfc.edu\Desktop\Object Dection\Pothole Detection\Data\images\train"
labels_path = r"C:\Users\bhagy\OneDrive - sfc.edu\Desktop\Object Dection\Pothole Detection\Data\labels\train"

# Connect to Azure Blob Storage
service = BlobServiceClient.from_connection_string(conn_str)
client = service.get_container_client(container_name)

# Create container if it doesn't exist
try:
    client.get_container_properties()
except Exception:
    client = service.create_container(container_name)
    print(f"Created container: {container_name}")

def upload_folder(local_folder, root_folder_name):
    all_files = []
    for root, _, files in os.walk(local_folder):
        for file in files:
            all_files.append(os.path.join(root, file))

    if not all_files:
        print(f"⚠️ No files found in {local_folder}")
        return 0

    print(f"Found {len(all_files)} files in {root_folder_name}")

    count = 0
    for path in all_files:
        blob_path = os.path.join(root_folder_name, os.path.relpath(path, local_folder)).replace("\\", "/")
        with open(path, "rb") as data:
            client.upload_blob(blob_path, data, overwrite=True)
        print(f"Uploaded: {blob_path}")
        count += 1
    return count

# Upload images and labels
count_images = upload_folder(images_path, "images/train")
count_labels = upload_folder(labels_path, "labels/train")

total_uploaded = count_images + count_labels
print(f"✅ Total uploaded files: {total_uploaded}")
print(f"Blob URL: https://{service.account_name}.blob.core.windows.net/{container_name}/")
