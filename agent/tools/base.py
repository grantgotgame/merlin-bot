"""BaseTool — interface that all agent tools implement."""


class BaseTool:
    """Base class for agent tools. Subclasses define name, description, parameters(), and execute()."""

    name: str = ""
    description: str = ""

    def parameters(self) -> dict:
        """Return JSON Schema dict describing the function parameters."""
        raise NotImplementedError

    def execute(self, **kwargs) -> str:
        """Execute the tool with given arguments. Return result as string."""
        raise NotImplementedError

    def to_ollama_schema(self) -> dict:
        """Convert to Ollama's tool format for /api/chat requests."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters(),
            },
        }
