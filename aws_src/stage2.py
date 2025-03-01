import json
import boto3
import pandas as pd

s3 = boto3.client("s3")


def lambda_handler(event, context):
    # Extract input from the event
    bucket_name = event["bucket_name"]
    file_key = event["file_key"]
    snp_dict = json.loads(event["snp_dict"])

    # Download the DNA CSV file from S3
    try:
        obj = s3.get_object(Bucket=bucket_name, Key=file_key)
        df = pd.read_csv(obj["Body"], comment='#', delimiter=',', names=["RSID", "CHROMOSOME", "POSITION", "GENOTYPE"])
    except Exception as e:
        print(f"Failed to read DNA CSV file: {e}")
        return {
            "statusCode": 500,
            "body": json.dumps({"error": "Failed to read DNA CSV file"})
        }

    # Find matching SNPs in the user's DNA data
    matched_data = df[df["RSID"].isin(snp_dict.keys())]

    # If no data matches, return an empty DataFrame
    if matched_data.empty:
        return {
            "statusCode": 200,
            "body": json.dumps({"message": "No matching data found"})
        }

    # Map detected SNPs to their descriptions
    matched_data["Trait"] = matched_data["RSID"].map(snp_dict)

    # Convert the result to JSON
    result = matched_data.to_json(orient="records")
    return {
        "statusCode": 200,
        "body": result
    }
