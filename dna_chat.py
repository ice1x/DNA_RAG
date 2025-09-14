import os
import json

import pandas as pd
from langchain_deepseek import ChatDeepSeek
from langchain.schema import HumanMessage
from typing import Optional, Dict


class DnaAnalysisClient:
    def __init__(self, api_key: str):
        """
        Initializes the DNA analysis client with the given API key.

        Args:
            api_key (str): The API key for accessing the DeepSeek model.
        """
        self.api_key: str = api_key
        self.llm: ChatDeepSeek = ChatDeepSeek(
            model="deepseek-r1:free",
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=2,
            api_key=self.api_key
        )

    def generate_snp_dict(self, query: str) -> Dict[str, Dict[str, str]]:
        """
        Generates an SNP dictionary based on the user's query using an LLM.

        Args:
            query (str): The user's query (e.g., "lactose tolerance").

        Returns:
            dict: A dictionary mapping RSID to their descriptions in the specified format.
        """
        prompt = f"""
        You are an expert in Python and genetics. Based on the user's query, create an SNP dictionary 
        mapping RSID to their descriptions.

        User query: {query}

        Return only the dictionary in this JSON format:
        {{
            "rs9786139": {{
                "gene": "MT-ND1",
                "chromosome": "MT",
                "position": 3307,
                "trait": "Ashkenazi Jewish ancestry (haplogroup K1a1b1a)"
            }}
        }}

        If there are no matching SNPs, return an empty dictionary: {{}}
        """

        try:
            response = self.llm([HumanMessage(content=prompt)]).content
            response = response.strip()

            # Try to parse the response as JSON
            try:
                snp_dict = json.loads(response)
                if not isinstance(snp_dict, dict):
                    raise ValueError("Response is not in the expected dictionary format.")
                return snp_dict

            except json.JSONDecodeError:
                print("Error: Response is not a valid JSON.")
                return {}

        except Exception as e:
            print(f"Unexpected error during LLM query or processing: {e}")
            return {}

    def process_dna_data(self, input_file: str, snp_dict: Dict[str, Dict[str, str]]) -> pd.DataFrame:
        """
        Filters DNA data from the input file based on the provided SNP dictionary.

        Args:
            input_file (str): Path to the CSV file containing DNA data.
            snp_dict (dict): SNP dictionary, format {"RSID": "Description"}.

        Returns:
            pd.DataFrame: Filtered data.
        """
        df = pd.read_csv(input_file, comment='#', delimiter=',', names=["RSID", "CHROMOSOME", "POSITION", "GENOTYPE"])
        matched_data = df[df["RSID"].isin(snp_dict.keys())]

        if matched_data.empty:
            return pd.DataFrame()

        matched_data["Trait"] = matched_data["RSID"].map(snp_dict)
        return matched_data

    def process_with_second_llm(self, filtered_data: pd.DataFrame, query: str) -> str:
        """
        Passes the filtered data to the LLM for analysis.

        Args:
            filtered_data (pd.DataFrame): SNP data.
            query (str): The original user query.

        Returns:
            str: Model's response.
        """
        data_str = filtered_data.to_string(index=False)

        prompt = f"""
        The DNA data has been filtered based on the query: {query}
        {data_str}

        Analyze the data and provide a response to the user.
        """

        response = self.llm([HumanMessage(content=prompt)]).content
        return response.strip()

    def run_analysis(self, query: str, input_file: str) -> Optional[str]:
        """
        Run the entire DNA analysis process.

        Args:
            query (str): The user's query.
            input_file (str): Path to the DNA CSV file.

        Returns:
            str: LLM's response to the analyzed data, or None if no response.
        """
        print("Generating SNP dictionary...")
        snp_dict = self.generate_snp_dict(query)

        if not snp_dict:
            print("No matching SNPs found.")
            return None

        print("SNP dictionary:\n", snp_dict)

        print("Processing DNA data...")
        result = self.process_dna_data(input_file, snp_dict)

        if not result.empty:
            print("Sending data to LLM...")
            llm_response = self.process_with_second_llm(result, query)
            print("\nLLM response:")
            print(llm_response)
            return llm_response
        else:
            print("No matching data found.")
            return None


def main():
    api_key = os.environ.get("API_KEY")
    if not api_key:
        print("API key is missing.")
        return

    client = DnaAnalysisClient(api_key)

    user_query = input("Enter a query (e.g., 'lactose tolerance'): ").lower()
    input_file = input("Enter the path to the DNA CSV file: ")

    client.run_analysis(user_query, input_file)


if __name__ == "__main__":
    main()
