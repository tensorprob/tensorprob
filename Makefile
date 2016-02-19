# This Makefile contains code from https://github.com/rootpy/root_numpy/blob/master/Makefile

DOCS=$(wildcard docs/sources/*.md)
ifeq ($(UNAME_S),Darwin)
	OPEN := open
else
	OPEN := xdg-open
endif
INTERACTIVE := $(shell ([ -t 0 ] && echo 1) || echo 0)

test:
	nosetests -s -v ./tests

test-coverage:
	nosetests -s -v -a '!slow' --with-coverage --cover-erase --cover-branches --cover-html --cover-html-dir=cover ./tests
	@if [ "$(INTERACTIVE)" -eq "1" ]; then \
		$(OPEN) cover/index.html; \
	fi;

format:
	yapf -i -r .

docs: docs/mkdocs.yml ${DOCS}
	cd docs/ && mkdocs build

publish:
	@ghp-import -n docs/site && git push -fq https://${GH_TOKEN}@github.com/${TRAVIS_REPO_SLUG}.git gh-pages > /dev/null

.PHONY: docs
