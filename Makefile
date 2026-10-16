.PHONY: data data-small features test lint all clean

data:
	python -m aml_lab.generate

data-small:
	python -m aml_lab.generate --small

features: data
	python -m aml_lab.features

test:
	pytest

lint:
	ruff check src/ tests/

all: features test

clean:
	rm -rf artifacts/data artifacts/features artifacts/data_card.md
