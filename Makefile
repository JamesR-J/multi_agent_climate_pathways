
IMAGE_TAG ?= cuda
IMAGE_NAME ?= multi_agent_climate_pathways
# API_KEY_FILE := api_key.txt
# WANDB_API_KEY := $(shell cat $(API_KEY_FILE))

.PHONY: build-docker
build-docker:
	docker buildx build --tag $(IMAGE_NAME):$(IMAGE_TAG) -f Dockerfile .

.PHONY: build-singularity
build-singularity: build-docker
	docker save $(IMAGE_NAME):$(IMAGE_TAG) -o image.tar
	singularity build $(IMAGE_NAME)-$(IMAGE_TAG).sif docker-archive://image.tar
	rm image.tar

.PHONY: clean-singularity
clean-singularity:
	rm *.sif

.PHONY: clean
clean:
	rm -rf build .cache