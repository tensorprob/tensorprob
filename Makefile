DOCS=$(wildcard docs/sources/*.md)

test:
	nosetests

format:
	yapf -i -r .

docs: docs/mkdocs.yml ${DOCS}
	cd docs/ && mkdocs build

publish:
	@ghp-import -n docs/site && git push -fq https://${GH_TOKEN}@github.com/${TRAVIS_REPO_SLUG}.git gh-pages > /dev/null
	cp -r docs/site/* _deploy/
	git config --global user.email ""
	git config --global user.name "Automatic travis commit"

.PHONY: docs
