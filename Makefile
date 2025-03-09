.PHONY: test check_format format check_typing fix clean

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +;
	find . -type d -name .pytest_cache -exec rm -rf {} +;
	find . -type d -name .mypy_cache -exec rm -rf {} +;
	find . -type d -name .ruff_cache -exec rm -rf {} +;
	find . -type f -name '*.pyc' -exec rm -f {} +;
	find . -type f -name '*.pyo' -exec rm -f {} +;
	find . -type d -name build -exec rm -rf {} +;
	find . -type d -name '*.egg-info' -exec rm -rf {} +;

install:
	pip install -e .[dev]

check_format:
	black --check .
	ruff check .

format:
	black .
	ruff check --fix .

check_typing:
	mypy src

fix: format check_typing

test:
	pytest -v ./tests