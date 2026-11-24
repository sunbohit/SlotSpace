PYTHON ?= python
WORK ?= runs/reproduce-v0
RESOURCES ?= ../Resource
THROUGH ?= M8
REPRO_ARGS ?=

.PHONY: reproduce-v0 test dist

reproduce-v0:
	"$(PYTHON)" -m slotspace.reproduce --work "$(WORK)" --resources "$(RESOURCES)" --through "$(THROUGH)" $(REPRO_ARGS)

test:
	"$(PYTHON)" -m pytest -q

dist:
	"$(PYTHON)" -m build --no-isolation
	"$(PYTHON)" -m twine check --strict dist/*
	"$(PYTHON)" tools/check_dist.py --dist dist
