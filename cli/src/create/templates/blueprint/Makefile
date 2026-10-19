# The image is <DOCKER_REGISTRY>/<REPO>:<VERSION>; set them on the command line
# (make push DOCKER_REGISTRY=hub.bubbleran.com REPO=orama/__BLUEPRINT_NAME__) or let
# `bat build` / `bat push` pass them.
DOCKER_REGISTRY ?=
REPO ?= __BLUEPRINT_NAME__
# The git tag or commit. Outside a git checkout there is none: the image is
# tagged `dev` and the agent card keeps its own version.
VERSION ?= $(shell git describe --tags --always --dirty 2>/dev/null)
IMAGE_TAG := $(if $(DOCKER_REGISTRY),$(DOCKER_REGISTRY)/)$(REPO):$(or $(VERSION),dev)

# Without a registry `docker push` would send the image to Docker Hub.
ifneq ($(filter push,$(MAKECMDGOALS)),)
ifeq ($(DOCKER_REGISTRY),)
$(error Set DOCKER_REGISTRY to push: make push DOCKER_REGISTRY=hub.bubbleran.com)
endif
endif

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
