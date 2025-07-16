from google.cloud import storage
from dotenv import load_dotenv
import os

load_dotenv()
GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME")

storage_client = storage.Client()

def upload_to_gcs(file_bytes: bytes, destination_blob_name: str) -> str:
    bucket = storage_client.bucket(GCS_BUCKET_NAME)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_string(file_bytes, content_type="application/pdf")
    return blob.public_url

def generate_signed_url(blob_name: str, expiration_minutes: int = 15) -> str:
    bucket = storage_client.bucket(GCS_BUCKET_NAME)
    blob = bucket.blob(blob_name)

    url = blob.generate_signed_url(
        expiration=expiration_minutes * 60,
        method="GET",
        version="v4",
        response_disposition=f'attachment; filename="{blob_name.split("/")[-1]}"'
    )
    return url
