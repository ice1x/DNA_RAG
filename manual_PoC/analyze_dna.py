"""
For: https://www.snpedia.com/
"""
import pandas as pd

# Load your MyHeritage raw DNA data
file_path = "~/Documents/MedAnalyze/ilia_DNA/MyHeritage_raw_dna_data.csv"  # Adjust if needed
df = pd.read_csv(file_path, comment='#', delimiter=',', names=["RSID", "CHROMOSOME", "POSITION", "GENOTYPE"])


def extract_data(keys_to_extract):
    # Find matching SNPs in the user's DNA data
    user_metabolism_snps = df[df["RSID"].isin(keys_to_extract.keys())]

    # Map detected SNPs to metabolism traits
    user_metabolism_snps["Trait"] = user_metabolism_snps["RSID"].map(keys_to_extract)

    # Save results to a file
    output_file = "../metabolism_results.csv"
    user_metabolism_snps.to_csv(output_file, index=False)

    # Print results
    print("Detected metabolism-related SNPs:")
    print(user_metabolism_snps)
    print(f"\nResults saved to {output_file}")

