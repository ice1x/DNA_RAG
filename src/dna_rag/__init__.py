"""DNA RAG -- Genetic analysis pipeline powered by LLMs.

The engine is **LLM-agnostic** -- each pipeline step can use a different
provider.  Example with per-step LLM selection::

    from pathlib import Path
    from dna_rag import DNAAnalysisEngine, Settings
    from dna_rag.llm.deepseek import DeepSeekProvider
    from dna_rag.llm.openai_compat import OpenAICompatProvider
    from dna_rag.cache import InMemoryCache

    snp_settings = Settings()                 # reads DNA_RAG_LLM_* env vars
    interp_settings = Settings(               # override for step 2
        llm_provider="openai_compat",
        llm_api_key="sk-...",
        llm_model="gpt-4o-mini",
        llm_base_url="https://api.openai.com/v1",
    )

    engine = DNAAnalysisEngine(
        snp_llm=DeepSeekProvider(snp_settings),
        interpretation_llm=OpenAICompatProvider(interp_settings),
        cache=InMemoryCache(),
    )
    result = engine.analyze("lactose tolerance", Path("my_dna.csv"))
    print(result.interpretation)
"""

from dna_rag._version import __version__
from dna_rag.config import Settings
from dna_rag.engine import DNAAnalysisEngine

__all__ = ["__version__", "DNAAnalysisEngine", "Settings"]
