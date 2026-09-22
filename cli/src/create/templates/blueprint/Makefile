DOCKER_REGISTRY ?= INSERT_YOUR_DOCKER_REGISTRY_HERE
REPO ?= YOUR_REPOSITORY/__BLUEPRINT_NAME__
VERSION ?= $(shell git describe --tags --always --abbrev --dirty)
IMAGE_TAG := $(DOCKER_REGISTRY)/$(REPO):$(VERSION)

# Every directory holding a config.yaml is an agent, so `bat add agent` never
# has to edit this file.
AGENTS := $(patsubst %/config.yaml,%,$(wildcard */config.yaml))

.PHONY: build push clean $(AGENTS)

# Run ONE agent locally against its own config.yaml. Both halves matter: the
# SDK reads CONFIG_PATH at startup (the blueprint's ./config.yaml is not the
# agent's), and the name selects which agent the shared entrypoint runs.
#   make <agent>
$(AGENTS):
	CONFIG_PATH=$@/config.yaml uv run . $@

build:
	docker build $(if $(NO_CACHE),--no-cache) \
		--build-arg VERSION=$(VERSION) \
		--tag $(IMAGE_TAG) .

push: build
	docker push $(IMAGE_TAG)

clean:
	-docker rmi $(IMAGE_TAG)
