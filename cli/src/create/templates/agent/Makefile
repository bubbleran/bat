# The image is <DOCKER_REGISTRY>/<REPO>:<VERSION>; set them on the command line
# (make push DOCKER_REGISTRY=hub.bubbleran.com REPO=orama/__AGENT_NAME__) or let
# `bat build` / `bat push` pass them.
DOCKER_REGISTRY ?=
REPO ?= __AGENT_NAME__
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

.PHONY: build run push clean

build:
	@echo "Building __AGENT_NAME__ Docker image with tag: $(IMAGE_TAG) - Version: $(VERSION)"
	docker build $(if $(NO_CACHE),--no-cache) \
		--build-arg VERSION=$(VERSION) \
		--tag $(IMAGE_TAG) .
	@echo "__AGENT_NAME__ Docker image built successfully."

run:
	docker run --rm \
		-it \
		--env-file .env \
		--network host \
		$(IMAGE_TAG)

push:
	@echo "Pushing __AGENT_NAME__ Docker image with tag: $(IMAGE_TAG)"
	docker push $(IMAGE_TAG)
	@echo "__AGENT_NAME__ Docker image pushed successfully."

clean:
	docker rmi $(IMAGE_TAG) || true
