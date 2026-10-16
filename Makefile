.PHONY: data data-small features segment test lint all clean

data:
	python -m aml_lab.generate

data-small:
	python -m aml_lab.generate --small

features: data
	python -m aml_lab.features

segment: features
	python -m aml_lab.segment

test:
	pytest

lint:
	ruff check src/ tests/

all: segment test

clean:
	rm -rf artifacts/data artifacts/features artifacts/segments artifacts/figures artifacts/data_card.md
