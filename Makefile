.PHONY: install install-validation test validate figures check advisor-si111
install:
	python -m pip install -e ".[plots,test]"

install-validation:
	python -m pip install "pip==26.2.1"
	python -m pip install -r requirements-validation.txt
	python -m pip install -e . --no-deps

test:
	pytest -q

validate:
	python scripts/build_manifest.py --check
	python scripts/validate_archive.py

figures:
	python scripts/reproduce_figures.py --check-reference

check: validate test figures

advisor-si111:
	ccgrowth-advisor --structure cases/si_111_gpaw/Si_111.cif --template cases/si_111_gpaw/Si_111.cif --axis z --species Si --quiet-command
