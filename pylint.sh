#!/bin/sh
# See https://github.com/microsoft/vscode-python/issues/4236.
timeout 1m pipenv run pylint $@
