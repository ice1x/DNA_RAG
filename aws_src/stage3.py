import json
import os
from openai import OpenAI

# Initialize OpenAI client
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])


def lambda_handler(event, context):
    # Extract input from the event
    filtered_data = json.loads(event["filtered_data"])
    query = event["query"]

    # Convert the filtered data to a string
    data_str = json.dumps(filtered_data, indent=2)

    # Create the second prompt
    prompt = f"""
    The following DNA data was filtered based on the user's query: {query}
    {data_str}

    Analyze this data and provide insights or answer the user's query.
    """
    response = client.chat.completions.create(
        model="gpt-4",  # or "gpt-3.5-turbo"
        messages=[{"role": "user", "content": prompt}],
        max_tokens=500,
    )

    # Return the LLM response
    return {
        "statusCode": 200,
        "body": response.choices[0].message.content.strip()
    }
