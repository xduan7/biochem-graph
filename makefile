PYTHON := $(shell which python)
PROJ_DIR := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))
PYTHONPATH := $(PYTHONPATH):$(PROJ_DIR)

help:
	@echo "Please use \`make <target>' where <target> is one of"
	@echo "    install:    install dependencies (require poetry)"
	@echo "    pytest:     unit test all cases implemented in ./src/tests"
	@echo "    mypy:       perform typing checking for python file"
	@echo "    lint:       perform style checking for python file"
	@echo "    check:      perform typing and style checking for python files"

# environment:
# 	@conda env export > environment.yml

install:
	@echo "installing dependencies with poetry ..."
	@poetry install
	@echo "[WARNING] please install Anaconda dependencies with following command: "
	@echo "	- conda env update -f=environment.yml"

pytest:
	@echo ${PYTHONPATH}
	@echo "unit testing ..."
	@pytest || true

mypy:
	@echo "checking source code typing with mypy ..."
	@mypy src --config-file mypy.ini  || true

lint:
	@echo "checking source code style with flake8 ..."
	@flake8 src || true
	@echo "checking source code style with pylint ..."
	@pylint --rcfile pylint.rc src || true

check:
	@$(MAKE) mypy
	@$(MAKE) lint


