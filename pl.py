import os
import subprocess
import pandas as pd
from openai import OpenAI

# Initialize OpenAI client (replace with your API key)
client = OpenAI(api_key="your-openai-api-key")


# Function to generate Python script using LLM
def generate_python_script(query, csv_columns):
    prompt = f"""
    You are a Python expert. Write a Python script that processes a CSV file with the following columns: {csv_columns}.
    The script should perform the following task: {query}.
    The script should read the CSV file as input, process it, and save the results to a new CSV file named 'output.csv'.
    Return only the Python code, without any explanations or markdown formatting.
    """
    response = client.chat.completions.create(
        model="gpt-4",  # or "gpt-3.5-turbo"
        messages=[{"role": "user", "content": prompt}],
        max_tokens=500,
    )
    return response.choices[0].message.content.strip()


# Function to validate the generated Python script
def validate_script(script):
    try:
        # Try to compile the script to check for syntax errors
        compile(script, "<string>", "exec")
        return True
    except SyntaxError as e:
        print(f"Script validation failed: {e}")
        return False


# Function to run the Python script
def run_script(script, input_file):
    try:
        # Save the script to a temporary file
        with open("temp_script.py", "w") as f:
            f.write(script)

        # Run the script using subprocess
        result = subprocess.run(
            ["python", "temp_script.py", input_file],
            capture_output=True,
            text=True,
        )

        # Check for errors
        if result.returncode != 0:
            print(f"Script execution failed: {result.stderr}")
            return False
        return True
    except Exception as e:
        print(f"Error running script: {e}")
        return False


# Function to process the output file with a second prompt
def process_output_with_llm(query):
    with open("output.csv", "r") as f:
        output_content = f.read()

    prompt = f"""
    The following CSV file was generated after processing the input file:
    {output_content}
    Based on this data, answer the following query: {query}.
    """
    response = client.chat.completions.create(
        model="gpt-4",  # or "gpt-3.5-turbo"
        messages=[{"role": "user", "content": prompt}],
        max_tokens=500,
    )
    return response.choices[0].message.content.strip()


# Main pipeline
def main():
    # Step 1: Get query from user and allow CSV file upload
    query = input("Enter your query: ")
    input_file = input("Enter the path to your CSV file: ")

    # Read CSV columns to provide context to the LLM
    df = pd.read_csv(input_file)
    csv_columns = df.columns.tolist()

    # Step 2: Generate Python script using LLM
    max_retries = 3
    for attempt in range(max_retries):
        print(f"Generating script (attempt {attempt + 1})...")
        script = generate_python_script(query, csv_columns)
        print("Generated script:\n", script)

        # Step 3: Validate the script
        if validate_script(script):
            print("Script is valid.")
            break
        else:
            print("Script is invalid. Retrying...")
    else:
        print("Failed to generate a valid script after multiple attempts.")
        return

    # Step 4: Run the Python script
    if run_script(script, input_file):
        print("Script executed successfully.")
    else:
        print("Script execution failed.")
        return

    # Step 5: Process the output file with a second prompt
    final_answer = process_output_with_llm(query)

    # Step 6: Return the final answer to the user
    print("\nFinal Answer:")
    print(final_answer)


# Run the pipeline
if __name__ == "__main__":
    main()