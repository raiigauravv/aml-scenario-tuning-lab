.PHONY: data data-small features segment scenarios tuning model test lint all clean

data:
	python -m aml_lab.generate

data-small:
	python -m aml_lab.generate --small

features: data
	python -m aml_lab.features

segment: features
	python -m aml_lab.segment

scenarios: data
	python -m aml_lab.scenarios

tuning: segment scenarios
	python -m aml_lab.tuning

model: tuning
	python -m aml_lab.prioritize

test:
	pytest

lint:
	ruff check src/ tests/

all: model test

clean:
	rm -rf artifacts/data artifacts/features artifacts/segments artifacts/figures artifacts/scenarios artifacts/metrics artifacts/models artifacts/data_card.md
