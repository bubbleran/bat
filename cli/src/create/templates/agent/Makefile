DOCKER_REGISTRY ?= INSERT_YOUR_DOCKER_REGISTRY_HERE
REPO ?= YOUR_REPOSITORY/__AGENT_NAME__
VERSION ?= $(shell git describe --tags --always --abbrev --dirty)
IMAGE_TAG := $(DOCKER_REGISTRY)/$(REPO):$(VERSION)

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
