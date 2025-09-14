class FakeLLM:
    """Deterministic LLM stub returning predefined responses."""

    def __init__(self, responses):
        self.responses = responses
        self.calls = 0

    def __call__(self, messages):
        content = self.responses[self.calls]
        self.calls += 1

        class Result:
            def __init__(self, content):
                self.content = content

        return Result(content)
