from __future__ import annotations

from pathlib import Path
from typing import Dict

import json
import pandas as pd
from pandas import DataFrame
from langchain_deepseek import ChatDeepSeek
from langchain.schema import HumanMessage


class ChatDNA:
    """High level interface for asking questions about DNA data.

    The class uses an LLM both to retrieve relevant SNP identifiers for a
    question and to interpret the user's DNA with respect to those SNPs.
    Loaded DNA files are cached so they can be reused between calls without
    re-reading them from disk.
    """

    def __init__(self, api_key: str) -> None:
        self.llm: ChatDeepSeek = ChatDeepSeek(
            model="deepseek-r1:free",
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=2,
            api_key=api_key,
        )
        self._dna_cache: Dict[Path, DataFrame] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def ask(self, question: str, dna_file: Path) -> str:
        """Answer *question* using the DNA information from *dna_file*.

        Parameters
        ----------
        question: str
            The user's question, e.g. "lactose tolerance".
        dna_file: Path
            Path to a CSV file in the 23andMe/AncestryDNA format.
        """

        df = self._load_dna(dna_file)
        snp_dict = self._get_snp_dict(question)
        if not snp_dict:
            return "No relevant SNPs found."

        filtered = self._filter_snps(df, snp_dict)
        if filtered.empty:
            return "No matching variants found in the provided DNA file."

        return self._interpret(filtered, question)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _get_snp_dict(self, question: str) -> Dict[str, Dict[str, str]]:
        """Query the LLM for SNP identifiers related to *question*.

        The LLM is expected to return a JSON dictionary mapping RSIDs to
        metadata describing the SNP.
        """

        prompt = f"""
        You are an expert in genetics. Based on the user's question, create
        a dictionary describing relevant SNPs in JSON format.

        Question: {question}

        The dictionary should look like:
        {{
            "rs123": {{
                "gene": "GENE", "chromosome": "1", "position": 1234,
                "trait": "Associated trait"
            }}
        }}
        Return an empty dictionary ({{}}) if nothing matches.
        """

        try:
            response = self.llm([HumanMessage(content=prompt)]).content.strip()
            snp_dict = json.loads(response)
            if isinstance(snp_dict, dict):
                return snp_dict  # type: ignore[return-value]
        except Exception:
            pass
        return {}

    def _load_dna(self, dna_file: Path) -> DataFrame:
        """Load *dna_file* into a :class:`~pandas.DataFrame` with caching."""

        if dna_file not in self._dna_cache:
            df = pd.read_csv(
                dna_file,
                comment="#",
                delimiter=",",
                names=["RSID", "CHROMOSOME", "POSITION", "GENOTYPE"],
            )
            self._dna_cache[dna_file] = df
        return self._dna_cache[dna_file]

    def _filter_snps(self, df: DataFrame, snp_dict: Dict[str, Dict[str, str]]) -> DataFrame:
        """Return rows from *df* whose RSIDs appear in *snp_dict*.

        The returned frame contains the original DNA information merged with
        the metadata from the SNP dictionary.
        """

        if not snp_dict:
            return DataFrame()

        matched = df[df["RSID"].isin(snp_dict.keys())].copy()
        if matched.empty:
            return matched

        details = (
            pd.DataFrame.from_dict(snp_dict, orient="index")
            .reset_index()
            .rename(columns={"index": "RSID"})
        )
        return matched.merge(details, on="RSID", how="left")

    def _interpret(self, df: DataFrame, question: str) -> str:
        """Ask the LLM to interpret *df* in the context of *question*."""

        data_str = df.to_string(index=False)
        prompt = f"""
        The following SNP data was extracted from a user's DNA file based on
        the question: {question}

        {data_str}

        Provide a concise interpretation for the user.
        """

        return self.llm([HumanMessage(content=prompt)]).content.strip()
