from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import hashlib
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
        self._dna_cache: Dict[Path, Tuple[str, DataFrame]] = {}
        self._snp_cache: Dict[str, Dict[str, Dict[str, str]]] = {}
        self._answer_cache: Dict[str, Dict[str, str]] = {}

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

        file_hash = self._hash_file(dna_file)
        df = self._load_dna(dna_file, file_hash)

        answers_for_file = self._answer_cache.setdefault(file_hash, {})
        if question in answers_for_file:
            return answers_for_file[question]

        snp_dict = self._snp_cache.get(question)
        if snp_dict is None:
            snp_dict = self._get_snp_dict(question)
            self._snp_cache[question] = snp_dict
        if not snp_dict:
            answer = "No relevant SNPs found."
            answers_for_file[question] = answer
            return answer

        filtered = self._filter_snps(df, snp_dict)
        if filtered.empty:
            answer = "No matching variants found in the provided DNA file."
            answers_for_file[question] = answer
            return answer

        answer = self._interpret(filtered, question)
        answers_for_file[question] = answer
        return answer

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

    def _hash_file(self, dna_file: Path) -> str:
        """Return a hash of *dna_file* to detect content changes."""

        return hashlib.sha256(dna_file.read_bytes()).hexdigest()

    def _load_dna(self, dna_file: Path, file_hash: str) -> DataFrame:
        """Load *dna_file* into a :class:`~pandas.DataFrame` with caching."""

        cached = self._dna_cache.get(dna_file)
        if cached is None or cached[0] != file_hash:
            df = pd.read_csv(
                dna_file,
                comment="#",
                delimiter=",",
                names=["RSID", "CHROMOSOME", "POSITION", "GENOTYPE"],
            )
            self._dna_cache[dna_file] = (file_hash, df)
        return self._dna_cache[dna_file][1]

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
