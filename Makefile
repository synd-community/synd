.PHONY: install fmt test

setup:
	python3.11 -m venv .venv &&
	. .venv/bin/activate && pip install --upgrade pip

install:
	pip install -e .[all]

fmt:
	ruff check --fix

test:
	pytest --cov=atlas tests/

download-samples:
	mkdir -p data/breast-cancer &&\
	wget https://archive.ics.uci.edu/static/public/14/breast+cancer.zip -O data/breast-cancer/breast+cancer.zip &&\
	unzip -d data/breast-cancer data/breast-cancer/breast+cancer.zip
