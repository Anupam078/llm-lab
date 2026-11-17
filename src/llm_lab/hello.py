import os
import platform
from importlib.metadata import version, PackageNotFoundError


def describe_environment() -> dict[str, str]:
    """Return facts about the runtime. Never returns secret values."""
    return {
        "python": platform.python_version(),
        "api_key_configured": str("GEMINI_API_KEY" in os.environ),
    }

def describe_dependencies(package: str="pytest") -> str|None:
    try:
        return version(package)
    except PackageNotFoundError:
        return None

def main() -> None:
    for key, value in describe_environment().items():
        print(f"{key}: {value}")

    print(f"pytest: {describe_dependencies()}")

if __name__ == "__main__":
    main()