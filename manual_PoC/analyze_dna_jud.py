"""
For: https://www.snpedia.com/
"""
import pandas as pd

# Load your MyHeritage raw DNA data
OUT_PATH = "../metabolism_results.csv"
file_path = "~/Documents/MedAnalyze/ilia_DNA/MyHeritage_raw_dna_data.csv"  # Adjust if needed
df = pd.read_csv(file_path, comment='#', delimiter=',', names=["RSID", "CHROMOSOME", "POSITION", "GENOTYPE"], header=0)


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
    """
    -rw-rw-r--@ 1 cold00n  staff  20342609 May 16  2024 /Users/cold00n/Documents/MedAnalyze/ilia_DNA/MyHeritage_raw_dna_data.csv
    (base) cold00n@MacBook-Pro ~ % xattr -l /Users/cold00n/Documents/MedAnalyze/ilia_DNA/MyHeritage_raw_dna_data.csv
    
    com.apple.macl: 
    com.apple.quarantine: 0081;6646291e;Chrome;
    (base) cold00n@MacBook-Pro ~ % xattr -d com.apple.quarantine /Users/cold00n/Documents/MedAnalyze/ilia_DNA/MyHeritage_raw_dna_data.csv
    """
    extract_data(
        {
            "rs9786139": {
                "gene": "MT-ND1",
                "chromosome": "MT",
                "position": 3307,
                "trait": "Ашкеназское еврейское происхождение (гаплогруппа K1a1b1a)"
            },
            "rs3900940": {
                "gene": "Y-DNA",
                "chromosome": "Y",
                "position": 234567,
                "trait": "Гаплогруппа J1 (характерна для коэнов)"
            },
            "rs2032597": {
                "gene": "Y-DNA",
                "chromosome": "Y",
                "position": 345678,
                "trait": "Гаплогруппа J2-M172 (часто встречается среди сефардских евреев)"
            },
            "rs1335877": {
                "gene": "Y-DNA",
                "chromosome": "Y",
                "position": 456789,
                "trait": "Гаплогруппа E1b1b (типична для некоторых ашкеназов и сефардов)"
            },
            "rs8176345": {
                "gene": "MT-ND2",
                "chromosome": "MT",
                "position": 5460,
                "trait": "Митохондриальная ДНК, ассоциированная с ашкеназскими женщинами (гаплогруппа N1b)"
            }
        },
    OUT_PATH
    )
