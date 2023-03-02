docker_build:
	docker build \
		--build-arg SRC_DIR=/pandemic_calc \
		-t mond/pandemic-calc \
		-f Dockerfile \
		.

docker_run:
	docker run \
		--gpus all \
		--rm \
		-it \
		-v ${shell pwd}:/pandemic_calc \
		-v /home/mond/datasets/pandemic:/dataset \
		-p 5000:5000 \
		-e "DATASET_DIR=/dataset" \
		-e "TF_CPP_MIN_LOG_LEVEL=2" \
		mond/pandemic-calc \
		bash

gen_dataset:
	@python ./src/dataset_gen.py \
	    ${DATASET_DIR}/resources/pandemic_cards \
	    ${DATASET_DIR}/resources/backgrounds \

show_model:
	@python ./src/show_model.py

train:
	@python ./src/train.py

evaluate:
	@python ./src/evaluate.py

test:
	@pytest ./src/tests

rest_service_run:
	@python ./src/predictor_service.py

reformat:
	@isort --line-length 100 .
	@black --line-length 100 .

clean:
	@find . -name "__pycache__" | xargs rm -rf
	@find . -name ".pytest_cache" | xargs rm -rf
	@find . -name ".mypy_cache" | xargs rm -rf
