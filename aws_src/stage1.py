import json
import os
from openai import OpenAI

# Initialize OpenAI client
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])


def lambda_handler(event, context):
    # Extract query from the event
    query = event["query"]

    # Generate the SNP dictionary
    prompt = f"""
    You are a Python expert. Based on the user's query, generate a dictionary of SNPs (or other filtering criteria) that can be used to process a DNA CSV file. The dictionary should map RSIDs to their corresponding descriptions (e.g., traits or conditions).

    The user's query is: {query}

    Return only the dictionary in the following format:
    {{
        "RSID1": "Description1",
        "RSID2": "Description2",
        ...
    }}

    For example, if the user asks for "lactose tolerance," return:
    {{
        "rs4988235": "Lactose tolerance (Europe)"
    }}

    If no relevant SNPs are found, return an empty dictionary: {{}}
    """
    response = client.chat.completions.create(
        model="gpt-4",  # or "gpt-3.5-turbo"
        messages=[{"role": "user", "content": prompt}],
        max_tokens=500,
    )

    # Extract the dictionary from the response
    try:
        snp_dict = eval(response.choices[0].message.content.strip())
        return {
            "statusCode": 200,
            "body": json.dumps(snp_dict)
        }
    except Exception as e:
        print(f"Failed to parse the dictionary from the response: {e}")
        return {
            "statusCode": 500,
            "body": json.dumps({"error": "Failed to generate SNP dictionary"})
        }
