PY      ?= .venv/bin/python
PORT    ?= 8000
SHOTS   ?= .screenshots
# Headless Chromium for `make check`. Override if Playwright's bundled browser works for you.
CHROMIUM ?= $(HOME)/.cache/ms-playwright/chromium-1234/chrome-linux64/chrome
# Locally extracted libnss3/libnspr4/libasound2 (no sudo on this machine).
CHROME_LIBS ?= $(HOME)/.local/lib/chromium-deps/root/usr/lib/x86_64-linux-gnu

.PHONY: all setup build verify check check-figures figures test serve clean cm

all: verify check check-figures

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

# Animated SVG figures: regenerate from their source, and check them in a headless browser
figures:
	$(PY) tools/figures/make_broadcast.py

check-figures: build
	LD_LIBRARY_PATH=$(CHROME_LIBS):$$LD_LIBRARY_PATH CHROMIUM=$(CHROMIUM) WB_DIST=dist FIG_FILE=1 $(PY) build/check_figures.py $(SHOTS)/figures

# If a server is already running on PORT (e.g. in another terminal), the fresh build is all you need:
# it serves dist/, so just refresh the browser. Use `make serve PORT=8001` for a second one.
serve: build
	@if $(PY) -c "import socket,sys; s=socket.socket(); sys.exit(s.connect_ex(('127.0.0.1', $(PORT))) != 0)"; then \
		echo "→ A server is already running on http://localhost:$(PORT)/ and now serves this fresh build: just refresh."; \
		echo "  (Want a separate one? make serve PORT=8001)"; \
	else \
		echo "→ http://localhost:$(PORT)/   (Ctrl+C to stop)"; \
		$(PY) -m http.server $(PORT) -d dist; \
	fi

clean:
	rm -rf dist $(SHOTS)
