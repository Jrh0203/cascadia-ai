.PHONY: fixture

TRANSACTION := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
SOURCE_ARCHIVE := $(TRANSACTION)/source.tar
SOURCE_MANIFEST := $(TRANSACTION)/source-manifest.json
ARCHIVE_VERIFIER := $(TRANSACTION)/archive-verify.py
WORKSPACE := $(TMPDIR)/r2-map-market-fixture
UV := UV_LINK_MODE=copy UV_PYTHON_INSTALL_DIR=$(UV_PROJECT_ENVIRONMENT)-python /opt/homebrew/bin/uv
GATE_PYTHONPATH := $(WORKSPACE)/python:$(WORKSPACE)

fixture:
	test ! -e "$(WORKSPACE)"
	PYTHONDONTWRITEBYTECODE=1 COPYFILE_DISABLE=1 /usr/bin/python3 -B "$(ARCHIVE_VERIFIER)" verify \
		--manifest "$(SOURCE_MANIFEST)" --archive "$(SOURCE_ARCHIVE)"
	mkdir -m 0700 "$(WORKSPACE)"
	umask 077 && COPYFILE_DISABLE=1 /usr/bin/tar -xf "$(SOURCE_ARCHIVE)" -C "$(WORKSPACE)"
	PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3 -B "$(ARCHIVE_VERIFIER)" verify-tree \
		--manifest "$(SOURCE_MANIFEST)" --repository "$(WORKSPACE)"
	cd "$(WORKSPACE)" && PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3 -B \
		tools/r2_map_w0_w5_source_manifest.py --repository . verify "$(SOURCE_MANIFEST)"
	cd "$(WORKSPACE)" && PYTHONPATH="$(GATE_PYTHONPATH)" $(UV) run --frozen --no-dev python \
		tools/r2_map_reference_panels.py --revision v1.1 verify \
		docs/v2/reports/r2-map-w0-reference-panel-manifest-v1.1.json
	cd "$(WORKSPACE)" && PYTHONPATH="$(GATE_PYTHONPATH)" $(UV) run --frozen --no-dev python \
		tools/r2_map_market_protocol_fixture.py
