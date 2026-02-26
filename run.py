"""
Thin wrapper: run the CLI entry point.
Use from project root: uv run python run.py  OR  uv run ml-classifier
"""
from ml_classifier.cli import main

if __name__ == "__main__":
    main()
