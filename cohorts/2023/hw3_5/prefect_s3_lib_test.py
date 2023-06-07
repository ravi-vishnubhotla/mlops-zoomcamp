from prefect_aws import S3Bucket

s3_bucket = S3Bucket.load("s3-bucket-example")
# print(s3_bucket.list_objects("prefect/data"))
s3_bucket.download_folder_to_path("prefect/data", "/opt/mlops/prefect-mlops-zoomcamp/data")

print("Done")