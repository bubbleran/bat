DOCKER_REGISTRY ?= INSERT_YOUR_DOCKER_REGISTRY_HERE
REPO ?= YOUR_REPOSITORY/__BLUEPRINT_NAME__
VERSION ?= $(shell git describe --tags --always --abbrev --dirty)
IMAGE_TAG := $(DOCKER_REGISTRY)/$(REPO):$(VERSION)

# Every directory holding a config.yaml is an agent, so `bat add agent` never
# has to edit this file.
AGENTS := $(patsubst %/config.yaml,%,$(wildcard */config.yaml))

.PHONY: build push clean $(AGENTS)

# Run ONE agent locally against its own config.yaml.
$(AGENTS):
	CONFIG_PATH=$@/config.yaml UV_ENV_FILE=$(wildcard .env) uv run . $@

# The image installs from the lockfile (`uv sync --frozen`), so there has to
# be one; commit it.
uv.lock: pyproject.toml
	uv lock

build: uv.lock
	docker build $(if $(NO_CACHE),--no-cache) \
		--build-arg VERSION=$(VERSION) \
		--tag $(IMAGE_TAG) .

push: build
	docker push $(IMAGE_TAG)

clean:
	-docker rmi $(IMAGE_TAG)
