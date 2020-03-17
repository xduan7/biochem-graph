.ONESHELL:
PYTHON := $(shell which python)
PROJ_DIR := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))
PYTHONPATH := $(PYTHONPATH):$(PROJ_DIR)
RED_N_BOLD_FONT := \033[1;31m
DEFAULT_FONT := \033[0m

help:
	@echo "Please use \`make <target>' where <target> is one of"
	@echo "    install:    install dependencies (require poetry)"
	@echo "    env         update the environment setup files"
	@echo "    pytest:     unit test all cases implemented in ./src/tests"
	@echo "    mypy:       perform typing checking for python file"
	@echo "    lint:       perform style checking for python file"
	@echo "    check:      perform typing and style checking for python files"


install:
	@echo "Installing dependencies with Poetry ..."
	@poetry install; \
	if [ $$? != 0 ]; then \
		echo "${RED_N_BOLD_FONT}[WARNING]${DEFAULT_FONT} Please install Poetry with 'pip install poetry'."; \
	fi
	@echo "Installing non-PyPi dependencies with Anaconda ..."
	@conda env update -f=environment.yml; \
	if [ $$? != 0 ]; then \
		echo "${RED_N_BOLD_FONT}[WARNING]${DEFAULT_FONT} Please install Anaconda dependencies with 'conda env update -f=environment.yml'."; \
	fi

env:
	@echo "Exporting non-PyPi dependencies with Anaconda ..."
	@conda env export > environment.yml; \
		if [ $$? != 0 ]; then \
			echo "${RED_N_BOLD_FONT}[WARNING]${DEFAULT_FONT} Please export Anaconda dependencies with 'conda env export > environment.yml' in the terminal."; \
		fi
	@echo "Exporting non-PyPi dependencies with Anaconda ..."
	@poetry check; \
	if [ $$? != 0 ]; then \
		echo "${RED_N_BOLD_FONT}[WARNING]${DEFAULT_FONT} Please install Poetry with 'pip install poetry'."; \
	fi

pytest:
	@echo ${PYTHONPATH}
	@echo "Performing unit testing with PyTest ..."
	@pytest || true

mypy:
	@echo "Checking source code typing with MyPy ..."
	@mypy src --config-file mypy.ini  || true

lint:
	@echo "Checking source code style with Flake8 ..."
	@flake8 src || true
	@echo "Checking source code style with Pylint ..."
	@pylint --rcfile pylint.rc src || true

check:
	@$(MAKE) mypy
	@$(MAKE) lint


