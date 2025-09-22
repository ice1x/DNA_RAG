# DNA_RAG

A minimal toolkit for asking questions about DNA files with the help of a
language model.

The project centres around the `ChatDNA` class (see `chat_dna.py`), which
retrieves relevant SNP identifiers for a question and interprets the user's DNA
data. A small command-line interface in `cli.py` exposes this functionality for
interactive use.

## Usage

Set the `API_KEY` environment variable with your DeepSeek API key and invoke the
CLI:

```bash
python cli.py --dna-file path/to/dna.csv --question "lactose tolerance"
```

You can also use `ChatDNA` directly from Python code:

```python
from pathlib import Path
from chat_dna import ChatDNA

chat = ChatDNA(api_key="your-key")
answer = chat.ask("lactose tolerance", Path("dna.csv"))
print(answer)
```

The older `DnaAnalysisClient` implementation has been removed in favour of the
more featureful `ChatDNA` interface.

