.PHONY: data data-small features segment scenarios tuning model papers results test lint all clean

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

papers: model
	python -m aml_lab.papers

results: papers
	python -m aml_lab.results

test:
	pytest

lint:
	ruff check src/ tests/ app/

all: results test

clean:
	rm -rf artifacts/data artifacts/features artifacts/segments artifacts/figures artifacts/scenarios artifacts/metrics artifacts/models artifacts/papers artifacts/data_card.md RESULTS.md
