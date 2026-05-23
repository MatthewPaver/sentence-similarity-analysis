PYTHON ?= python3
VENV ?= .venv
PYTHON_BIN := $(VENV)/bin/python
PIP_BIN := $(PYTHON_BIN) -m pip

.PHONY: venv install check-data critic test notebook

venv:
	$(PYTHON) -m venv $(VENV)

install: venv
	$(PIP_BIN) install --upgrade pip
	$(PIP_BIN) install -r requirements.txt
	$(PYTHON_BIN) -m ipykernel install --user --name=embeds --display-name="embeds"

check-data:
	test -s data.txt
	test -s response.ipynb

critic:
	$(PYTHON_BIN) retrieval_critic.py "polar bear fur"

test: check-data
	$(PYTHON) -m pytest tests -q

notebook: install
	$(PYTHON_BIN) -m jupyter lab response.ipynb
