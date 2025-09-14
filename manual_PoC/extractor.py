import pandas as pd


def process_dna_data(input_file, snp_dict):
    """
    Фильтрует данные DNA-файла по заданному словарю SNP.

    Args:
        input_file (str): Путь к файлу CSV с DNA-данными.
        snp_dict (dict): Словарь SNP, формат {"RSID": "Описание"}.

    Returns:
        pd.DataFrame: Отфильтрованные данные.
    """
    df = pd.read_csv(input_file, comment='#', delimiter=',', names=["RSID", "CHROMOSOME", "POSITION", "GENOTYPE"])
    matched_data = df[df["RSID"].isin(snp_dict.keys())]

    if matched_data.empty:
        return pd.DataFrame()

    matched_data["Trait"] = matched_data["RSID"].map(snp_dict)
    return matched_data


if __name__ == "__main__":
    file_path = "/Users/cold00n/Documents/MedAnalyze/ilia_DNA/MyHeritage_raw_dna_data.csv"  # Adjust if needed
    snp_dict = {
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
    }

    print(process_dna_data(file_path, snp_dict))
