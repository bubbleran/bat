KUBE_CONFIG := ~/.kube/config
DOCKER_REGISTRY ?= INSERT_YOUR_DOCKER_REGISTRY_HERE
VERSION ?= $(shell git describe --tags --always --abbrev --dirty)

PORT ?= 9900
REPO ?= YOUR_REPOSITORY/api
TAG ?= latest
IMAGE := $(DOCKER_REGISTRY)/$(REPO):$(TAG)

.PHONY: build run push clean

build:
	@echo "Building api Agent Docker image with tag: $(IMAGE) - Version: $(VERSION)"
	docker build $(if $(NO_CACHE),--no-cache) \
		--build-arg VERSION=$(VERSION) \
		--tag $(IMAGE) .
	@echo "api Agent Docker image built successfully."

run:
	docker run --rm \
		-it \
		--env-file .env \
		--network host \
		-p $(PORT):$(PORT) \
		-v $(KUBE_CONFIG):/root/.kube/config:ro \
		$(IMAGE)

push:
	@echo "Pushing api Agent Docker image with tag: $(IMAGE)"
	docker push $(IMAGE)
	@echo "api Agent Docker image pushed successfully."

clean:
	docker rmi $(IMAGE) || true
