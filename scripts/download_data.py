import os

from azure.storage.blob import BlobServiceClient
from dotenv import load_dotenv


load_dotenv()

conn_str = os.getenv("AZURE_CONN_STR")
container_name = os.getenv("AZURE_CONTAINER_NAME")
download_root = os.getenv("LOCAL_PATH", "data")

service = BlobServiceClient.from_connection_string(conn_str)
client = service.get_container_client(container_name)


def download_folder_from_azure(container_client, folder_prefix, local_download_dir):
    """Download all blobs under a folder prefix, preserving structure locally."""
    if not folder_prefix.endswith("/"):
        folder_prefix += "/"

    print(f"Downloading blobs with prefix '{folder_prefix}' ...")

    blobs = container_client.list_blobs(name_starts_with=folder_prefix)
    count = 0

    for blob in blobs:
        blob_client = container_client.get_blob_client(blob.name)
        local_path = os.path.join(
            local_download_dir, os.path.relpath(blob.name, folder_prefix)
        )
        os.makedirs(os.path.dirname(local_path), exist_ok=True)

        try:
            with open(local_path, "wb") as f:
                blob_client.download_blob().readinto(f)
            print(f"Downloaded: {blob.name} -> {local_path}")
            count += 1
        except Exception as e:
            print(f"Failed to download {blob.name}: {e}")

    if count == 0:
        print(f"No blobs found for prefix '{folder_prefix}'")
    else:
        print(f"Downloaded {count} files from '{folder_prefix}'")


if __name__ == "__main__":
    download_folder_from_azure(
        client, "images/train", os.path.join(download_root, "images/train")
    )
    download_folder_from_azure(
        client, "labels/train", os.path.join(download_root, "labels/train")
    )
    print(f"\nAll downloads complete. Files saved under: {download_root}")
    print(f"Blob URL base: https://{service.account_name}.blob.core.windows.net/{container_name}/")
