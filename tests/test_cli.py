import os
import sys
import subprocess
from pathlib import Path


def test_cli_ask(tmp_path):
    dna_file = tmp_path / "dna.csv"
    dna_file.write_text("rs1,1,111,AA\n")

    stub_dir = tmp_path / "stub"
    stub_dir.mkdir()
    stub_file = stub_dir / "langchain_deepseek.py"
    stub_file.write_text(
        '''
class ChatDeepSeek:
    def __init__(self, *args, **kwargs):
        self.calls = 0
    def __call__(self, messages):
        responses = [
            '{"rs1": {"gene": "GENE", "chromosome": "1", "position": 111}}',
            'cli answer'
        ]
        content = responses[self.calls]
        self.calls += 1
        class R:
            def __init__(self, c):
                self.content = c
        return R(content)
'''
    )

    env = os.environ.copy()
    env["API_KEY"] = "test"
    repo_root = Path(__file__).resolve().parents[1]
    env["PYTHONPATH"] = f"{stub_dir}{os.pathsep}{repo_root}"

    result = subprocess.run(
        [sys.executable, "-m", "cli", "--dna-file", str(dna_file), "--question", "hi"],
        capture_output=True,
        text=True,
        env=env,
        cwd=tmp_path,
        check=True,
    )

    assert "cli answer" in result.stdout
