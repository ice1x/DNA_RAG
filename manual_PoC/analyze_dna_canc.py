"""
For: https://www.snpedia.com/
"""
import pandas as pd

# Load your MyHeritage raw DNA data
OUT_PATH = "../metabolism_results.csv"
file_path = "~/Documents/MedAnalyze/ilia_DNA/MyHeritage_raw_dna_data.csv"  # Adjust if needed
df = pd.read_csv(file_path, comment='#', delimiter=',', names=["RSID", "CHROMOSOME", "POSITION", "GENOTYPE"])


def extract_data(keys_to_extract, output_file):
    # Find matching SNPs in the user's DNA data
    user_metabolism_snps = df[df["RSID"].isin(keys_to_extract.keys())]

    # Map detected SNPs to metabolism traits
    user_metabolism_snps["Trait"] = user_metabolism_snps["RSID"].map(keys_to_extract)

    # Save results to a file
    user_metabolism_snps.to_csv(output_file, index=False)

    # Print results
    print("Detected metabolism-related SNPs:")
    print(user_metabolism_snps)
    print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    extract_data(
        {
            "rs2290907": {
                "gene": "TMC6 (EVER1)",
                "chromosome": 17,
                "position": 7571756,
                "trait": "Immune response to HPV, susceptibility to warts"
            },
            "rs7208422": {
                "gene": "TMC8 (EVER2)",
                "chromosome": 17,
                "position": 7574032,
                "trait": "Immune response to HPV, susceptibility to warts"
            },
            "rs9275319": {
                "gene": "HLA-DQA1",
                "chromosome": 6,
                "position": 32611893,
                "trait": "Immune system regulation, HPV-related infections"
            },
            "rs3135391": {
                "gene": "HLA-DRB1",
                "chromosome": 6,
                "position": 32407620,
                "trait": "Immune response to infections, HPV susceptibility"
            },
            "rs6457617": {
                "gene": "HLA-DQB1",
                "chromosome": 6,
                "position": 32582574,
                "trait": "Immune regulation and response to HPV"
            }
        },
    OUT_PATH
    )
