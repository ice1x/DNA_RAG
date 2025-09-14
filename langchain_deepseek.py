class ChatDeepSeek:
    """Minimal stub of the ChatDeepSeek LLM for tests."""

    def __init__(self, *args, responses=None, **kwargs):
        self.responses = responses or []
        self.calls = 0

    def __call__(self, messages):
        if self.responses:
            content = self.responses[self.calls]
            self.calls += 1
        else:
            content = ""

        class R:
            def __init__(self, content):
                self.content = content

        return R(content)
