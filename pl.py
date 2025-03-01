import pandas as pd
from openai import OpenAI

# Initialize OpenAI client (replace with your API key)
client = OpenAI(api_key="your-openai-api-key")


# Function to generate SNP dictionary using LLM
def generate_snp_dict(query):
    """
    Generate a dictionary of SNPs based on the user's query using an LLM.

    Args:
        query (str): The user's query (e.g., "lactose tolerance").

    Returns:
        dict: A dictionary mapping RSIDs to their descriptions.
    """
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
        return snp_dict
    except:
        print("Failed to parse the dictionary from the response.")
        return {}


# Function to process DNA data using the SNP dictionary
def process_dna_data(input_file, snp_dict):
    """
    Process a DNA CSV file based on a dictionary of SNPs.

    Args:
        input_file (str): Path to the input CSV file.
        snp_dict (dict): Dictionary of SNPs to filter. Format: {"RSID": "Description"}.

    Returns:
        pd.DataFrame: Filtered DataFrame containing matched rows.
    """
    # Load the DNA CSV file
    df = pd.read_csv(input_file, comment='#', delimiter=',', names=["RSID", "CHROMOSOME", "POSITION", "GENOTYPE"])

    # Find matching SNPs in the user's DNA data
    matched_data = df[df["RSID"].isin(snp_dict.keys())]

    # If no data matches, return an empty DataFrame
    if matched_data.empty:
        return pd.DataFrame()

    # Map detected SNPs to their descriptions
    matched_data["Trait"] = matched_data["RSID"].map(snp_dict)

    return matched_data


# Function to pass filtered data to the second LLM request
def process_with_second_llm(filtered_data, query):
    """
    Pass the filtered data to a second LLM request for further analysis.

    Args:
        filtered_data (pd.DataFrame): Filtered DataFrame containing matched rows.
        query (str): The user's original query.

    Returns:
        str: The LLM's response.
    """
    # Convert the filtered data to a string
    data_str = filtered_data.to_string(index=False)

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
    return response.choices[0].message.content.strip()


# Main pipeline
def main():
    # Step 1: Get user query and input file path
    user_query = input("Enter your query (e.g., 'lactose tolerance'): ").lower()
    input_file = input("Enter the path to your DNA CSV file: ")

    # Step 2: Generate the SNP dictionary
    print("Generating SNP dictionary...")
    snp_dict = generate_snp_dict(user_query)

    if not snp_dict:
        print("No matching SNPs found for the query.")
        return

    print("Generated SNP dictionary:\n", snp_dict)

    # Step 3: Process the DNA data
    print("Processing DNA data...")
    result = process_dna_data(input_file, snp_dict)

    # Step 4: Pass the filtered data to the second LLM request
    if not result.empty:
        print("Passing filtered data to the second LLM request...")
        llm_response = process_with_second_llm(result, user_query)
        print("\nLLM Response:")
        print(llm_response)
    else:
        print("No matching data found.")


# Run the pipeline
if __name__ == "__main__":
    main()