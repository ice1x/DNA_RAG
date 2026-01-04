"""Enhanced command-line interface for ChatDNAEnhanced.

This CLI provides access to enhanced features including RAG, validation,
structured output, and polygenic risk scores.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from dna_rag.core.chat_dna_enhanced import ChatDNAEnhanced
from dna_rag.core.config import get_settings
from dna_rag.utils.polygenic_scores import PolygenicScoreCalculator


def main() -> None:
    """Entry point for the enhanced DNA-RAG CLI."""

    parser = argparse.ArgumentParser(
        description="Ask questions about your DNA using LLM with RAG and validation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic question
  python cli_enhanced.py --dna-file dna.csv --question "lactose tolerance"

  # Structured output as JSON
  python cli_enhanced.py --dna-file dna.csv --question "eye color" --json

  # Calculate polygenic risk score
  python cli_enhanced.py --dna-file dna.csv --prs alzheimers_risk

  # Interactive mode
  python cli_enhanced.py --dna-file dna.csv --interactive
        """,
    )

    parser.add_argument(
        "--dna-file",
        required=True,
        type=Path,
        help="Path to the DNA file (CSV or VCF)",
    )
    parser.add_argument(
        "--question",
        help="Question to ask about the DNA",
    )
    parser.add_argument(
        "--prs",
        metavar="SCORE_NAME",
        help="Calculate polygenic risk score (e.g., alzheimers_risk, type2_diabetes_risk)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Interactive mode for multiple questions",
    )
    parser.add_argument(
        "--no-rag",
        action="store_true",
        help="Disable RAG (use LLM only)",
    )
    parser.add_argument(
        "--no-validation",
        action="store_true",
        help="Disable SNP validation through dbSNP",
    )

    args = parser.parse_args()

    # Load configuration
    settings = get_settings()
    try:
        settings.validate_api_key()
    except ValueError as exc:
        parser.error(str(exc))

    # Initialize enhanced chat
    chat = ChatDNAEnhanced(
        api_key=settings.deepseek_api_key,
        use_vector_store=not args.no_rag,
        use_validation=not args.no_validation,
        vector_store_path=settings.get_vector_store_path(),
    )

    # Handle polygenic risk score
    if args.prs:
        calculate_prs(args.dna_file, args.prs, args.json)
        return

    # Handle single question
    if args.question:
        result = chat.ask(args.question, args.dna_file, return_structured=True)
        print_result(result, args.json)
        return

    # Interactive mode
    if args.interactive or not args.question:
        interactive_mode(chat, args.dna_file, args.json)


def print_result(result, as_json: bool = False) -> None:
    """Print interpretation result.

    Parameters
    ----------
    result : InterpretationResult
        The result to print.
    as_json : bool
        Whether to output as JSON.
    """
    if as_json:
        print(result.model_dump_json(indent=2))
    else:
        print("\n" + "=" * 60)
        print(f"Question: {result.question}")
        print("=" * 60)
        print(f"\n{result.interpretation}\n")
        print(f"Confidence: {result.confidence:.2f}")
        print(f"\nSNPs found: {len(result.snps_found)}")
        if result.snps_found:
            print("\nDetailed SNP information:")
            for snp in result.snps_found[:5]:  # Show first 5
                print(f"  - {snp.rsid}: {snp.genotype} ({snp.trait})")
            if len(result.snps_found) > 5:
                print(f"  ... and {len(result.snps_found) - 5} more")
        if result.sources:
            print(f"\nSources: {', '.join(result.sources)}")
        if result.caveats:
            print("\n⚠️  Important notes:")
            for caveat in result.caveats:
                print(f"  - {caveat}")
        print("\n" + "=" * 60 + "\n")


def calculate_prs(dna_file: Path, score_name: str, as_json: bool = False) -> None:
    """Calculate and display polygenic risk score.

    Parameters
    ----------
    dna_file : Path
        Path to DNA file.
    score_name : str
        Name of the score to calculate.
    as_json : bool
        Whether to output as JSON.
    """
    import pandas as pd

    # Load DNA data
    df = pd.read_csv(
        dna_file,
        comment="#",
        delimiter=",",
        names=["RSID", "CHROMOSOME", "POSITION", "GENOTYPE"],
    )

    # Calculate score
    calculator = PolygenicScoreCalculator()
    try:
        result = calculator.calculate(score_name, df)
    except ValueError as exc:
        print(f"Error: {exc}")
        print(f"\nAvailable scores: {', '.join(calculator.list_available_scores())}")
        return

    if as_json:
        print(result.model_dump_json(indent=2))
    else:
        print("\n" + "=" * 60)
        print(f"Polygenic Risk Score: {score_name}")
        print("=" * 60)
        print(f"\nRaw Score: {result.raw_score:.2f}")
        print(f"Normalized Score: {result.normalized_score:.2f}")
        print(f"Percentile: {result.percentile:.1f}%")
        print(f"\nSNPs matched: {result.snps_matched}/{result.snps_total}")
        print(f"\n{result.interpretation}")
        print("\n" + "=" * 60 + "\n")


def interactive_mode(chat: ChatDNAEnhanced, dna_file: Path, as_json: bool) -> None:
    """Run interactive question-answering mode.

    Parameters
    ----------
    chat : ChatDNAEnhanced
        Chat instance.
    dna_file : Path
        DNA file path.
    as_json : bool
        Whether to output as JSON.
    """
    print("\n" + "=" * 60)
    print("Interactive DNA Analysis Mode")
    print("=" * 60)
    print("Enter your questions (blank to quit)")
    print("Type 'history' to see conversation history")
    print("Type 'prs <score_name>' to calculate polygenic risk score")
    print("=" * 60 + "\n")

    while True:
        try:
            question = input("Question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not question:
            print("Goodbye!")
            break

        if question.lower() == "history":
            history = chat.get_conversation_history(dna_file)
            print(f"\nConversation history ({len(history)} messages):")
            for msg in history:
                print(f"  [{msg.role}]: {msg.content[:80]}...")
            print()
            continue

        if question.lower().startswith("prs "):
            score_name = question.split(maxsplit=1)[1]
            calculate_prs(dna_file, score_name, as_json)
            continue

        result = chat.ask(question, dna_file, return_structured=True)
        print_result(result, as_json)


if __name__ == "__main__":  # pragma: no cover
    main()
