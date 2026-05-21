"""Quick smoke-test: does DeepSeek call tools via llmstitch + OpenAIAdapter?"""
import asyncio
from llmstitch import Agent, tool
from llmstitch.providers.openai import OpenAIAdapter

API_KEY  = "sk-7e92a44822fe40108902b49707408efe"
BASE_URL = "https://api.deepseek.com/v1"
MODEL    = "deepseek-v4-pro"


@tool
def list_files(pattern: str) -> str:
    """List files matching a glob pattern in the current repository."""
    import glob
    matches = glob.glob(pattern, recursive=True)
    return "\n".join(matches) if matches else "(no matches)"


@tool
def read_file(path: str) -> str:
    """Read the contents of a file."""
    try:
        with open(path, encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return f"error: file not found: {path}"


async def main() -> None:
    adapter = OpenAIAdapter(api_key=API_KEY, base_url=BASE_URL)
    agent = Agent(
        provider=adapter,
        model=MODEL,
        system="You are a coding agent. Use tools to inspect the repository before answering.",
    )
    agent.tools.register(list_files)
    agent.tools.register(read_file)

    question = "What Python files are in the agent/ directory?"
    print(f"Q: {question}\n")

    history = await agent.run(question)
    print(f"A: {history[-1].content}")


if __name__ == "__main__":
    asyncio.run(main())
