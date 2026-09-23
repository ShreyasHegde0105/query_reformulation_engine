# Minimal LLM client stub for testing purposes

class LLMClient:
    """A placeholder LLM client.

    In production this would handle communication with an LLM provider.
    The test suite mocks ``generate``/``chat`` methods, so the implementation
    simply raises ``NotImplementedError``.
    """

    def generate(self, prompt: str) -> str:
        """Generate a response for the given *prompt*.

        The real client would call the provider's API. In the test environment
        this method is patched, so we raise ``NotImplementedError`` to make any
        accidental calls obvious.
        """
        raise NotImplementedError("LLMClient.generate is a stub and should be mocked in tests.")
