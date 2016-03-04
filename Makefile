# This Makefile contains code from https://github.com/rootpy/root_numpy/blob/master/Makefile

DOCS=$(wildcard docs/sources/*.md)
ifeq ($(UNAME_S),Darwin)
	OPEN := open
else
	OPEN := xdg-open
endif

test:
	nosetests -s -v ./tests

test-coverage:
	nosetests -s -v -a '!slow' --with-coverage --cover-erase --cover-branches --cover-html --cover-html-dir=cover ./tests

format:
	yapf -i -r .

docs:
	cd docs/ && make html

publish:
	@ghp-import -n docs/build/html && git push -fq https://${GH_TOKEN}@github.com/${TRAVIS_REPO_SLUG}.git gh-pages > /dev/null

.PHONY: docs
