export VENV := ./venv

.PHONY: install
install: python

.PHONY: python
python:  venv
	. $(VENV)/bin/activate && pip install -e .[dev]

venv:
	test -d $(VENV) || python3 -m venv $(VENV)
	. $(VENV)/bin/activate && pip install --upgrade pip setuptools wheel

.PHONY: clean
clean:
	-mv  *.log ./trsh/
	-mv  *.vtp ./trsh/
	-mv  *.dat ./trsh/

.PHONY: distclean
distclean:
	rm -rf $(VENV)/
	rm -rf .tox/
	rm -rf isct.egg-info/
	rm -rf cov_html/
