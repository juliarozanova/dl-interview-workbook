PY      ?= .venv/bin/python
PORT    ?= 8000
SHOTS   ?= .screenshots
# Headless Chromium for `make check`. Override if Playwright's bundled browser works for you.
CHROMIUM ?= $(HOME)/.cache/ms-playwright/chromium-1234/chrome-linux64/chrome
# Locally extracted libnss3/libnspr4/libasound2 (no sudo on this machine).
CHROME_LIBS ?= $(HOME)/.local/lib/chromium-deps/root/usr/lib/x86_64-linux-gnu

.PHONY: all setup build verify check test serve clean cm

all: verify check

setup:
	uv venv -q .venv
	uv pip install -q --python $(PY) -e .

build:
	$(PY) build/build.py

# Content gate: in-process checks, then the generated companion tests against the reference solutions.
verify: build
	$(PY) build/verify.py
	WB_IMPL=solutions $(PY) -m pytest

# Your own work: companion/exercises against the generated tests. Narrow it with T=companion/tests/ch13
test:
	$(PY) -m pytest $(T)

# Rebuild the vendored CodeMirror bundle (only needed when changing tools/codemirror/entry.js)
cm:
	cd tools/codemirror && npm install --silent && npm run -s build

check: build
	LD_LIBRARY_PATH=$(CHROME_LIBS):$$LD_LIBRARY_PATH CHROMIUM=$(CHROMIUM) $(PY) build/check_browser.py $(SHOTS)

serve: build
	@echo "→ http://localhost:$(PORT)/"
	$(PY) -m http.server $(PORT) -d dist

clean:
	rm -rf dist $(SHOTS)
