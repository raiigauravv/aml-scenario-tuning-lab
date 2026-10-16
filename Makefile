.PHONY: data test lint all clean

data:
	python -m aml_lab.generate

data-small:
	python -m aml_lab.generate --small

test:
	pytest

lint:
	ruff check src/ tests/

all: data test

clean:
	rm -rf artifacts/data artifacts/data_card.md
