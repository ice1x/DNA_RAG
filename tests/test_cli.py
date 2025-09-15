import sys
from unittest.mock import Mock

import cli


def test_cli_ask(tmp_path, monkeypatch, capsys):
    dna_file = tmp_path / "dna.csv"
    dna_file.write_text("rs1,1,111,AA\n")

    responses = [
        {
            "choices": [
                {"message": {"content": '{"rs1": {"gene": "GENE", "chromosome": "1", "position": 111}}'}}
            ]
        },
        {"choices": [{"message": {"content": "cli answer"}}]},
    ]
    call = {"i": 0}

    def fake_post(url, headers, json, timeout):
        data = responses[call["i"]]
        call["i"] += 1
        mock_resp = Mock()
        mock_resp.json.return_value = data
        mock_resp.raise_for_status.return_value = None
        return mock_resp

    monkeypatch.setattr("langchain_deepseek.requests.post", fake_post)
    monkeypatch.setenv("API_KEY", "test")
    monkeypatch.setattr(sys, "argv", [
        "cli",
        "--dna-file",
        str(dna_file),
        "--question",
        "hi",
    ])

    cli.main()
    captured = capsys.readouterr()
    assert "cli answer" in captured.out

