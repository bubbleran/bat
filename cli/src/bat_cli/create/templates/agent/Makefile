KUBE_CONFIG := ~/.kube/config
DOCKER_REGISTRY ?= INSERT_YOUR_DOCKER_REGISTRY_HERE
VERSION ?= $(shell git describe --tags --always --abbrev --dirty)

PORT ?= 9900
REPO ?= YOUR_REPOSITORY/__PROJECT_NAME__
TAG ?= latest
IMAGE := $(DOCKER_REGISTRY)/$(REPO):$(TAG)

.PHONY: build run push clean

build:
	@echo "Building __PROJECT_NAME__ Agent Docker image with tag: $(IMAGE) - Version: $(VERSION)"
	docker build $(if $(NO_CACHE),--no-cache) \
		--build-arg VERSION=$(VERSION) \
		--tag $(IMAGE) .
	@echo "__PROJECT_NAME__ Agent Docker image built successfully."

run:
	docker run --rm \
		-it \
		--env-file .env \
		--network host \
		-p $(PORT):$(PORT) \
		-v $(KUBE_CONFIG):/root/.kube/config:ro \
		$(IMAGE)

push:
	@echo "Pushing __PROJECT_NAME__ Agent Docker image with tag: $(IMAGE)"
	docker push $(IMAGE)
	@echo "__PROJECT_NAME__ Agent Docker image pushed successfully."

clean:
	docker rmi $(IMAGE) || true
