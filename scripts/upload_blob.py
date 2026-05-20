import os

from azure.storage.blob import BlobServiceClient
from dotenv import load_dotenv


load_dotenv()

conn_str = os.getenv("AZURE_CONN_STR")
container_name = os.getenv("AZURE_CONTAINER_NAME")
images_path = os.getenv("IMAGES_PATH", "data/images/train")
labels_path = os.getenv("LABELS_PATH", "data/labels/train")

service = BlobServiceClient.from_connection_string(conn_str)
client = service.get_container_client(container_name)

try:
    client.get_container_properties()
except Exception:
    client = service.create_container(container_name)
    print(f"Created container: {container_name}")


def upload_folder(local_folder, root_folder_name):
    """Upload all files in a local folder to Azure Blob Storage."""
    all_files = [
        os.path.join(root, f)
        for root, _, files in os.walk(local_folder)
        for f in files
    ]

    if not all_files:
        print(f"No files found in {local_folder}")
        return 0

    print(f"Found {len(all_files)} files in {root_folder_name}")

    count = 0
    for path in all_files:
        blob_path = (
            os.path.join(root_folder_name, os.path.relpath(path, local_folder))
            .replace("\\", "/")
        )
        with open(path, "rb") as f:
            client.upload_blob(blob_path, f, overwrite=True)
        print(f"Uploaded: {blob_path}")
        count += 1

    return count


if __name__ == "__main__":
    count_images = upload_folder(images_path, "images/train")
    count_labels = upload_folder(labels_path, "labels/train")
    print(f"Total uploaded files: {count_images + count_labels}")
    print(f"Blob URL: https://{service.account_name}.blob.core.windows.net/{container_name}/")
