JOB_NAME = softmax_6

PACKAGE_NAME=kaggle_nflbowl_yards
FILENAME=model

PYTHON_VERSION=3.7
FRAMEWORK=scikit-learn
RUNTIME_VERSION=1.15

# path to the file to upload to GCP (the path to the file should be absolute or should match the directory where the make command is ran)
# replace with your local path to the `train_1k.csv` and make sure to put the path between quotes
PATH_DATA_PROCESSED="kaggle_nflbowl_yards/data/train_processed.csv"
PATH_TARGET_PROCESSED="kaggle_nflbowl_yards/data/target_processed.csv"

# bucket directory in which to store the uploaded file (`data` is an arbitrary name that we choose to use)
BUCKET_FOLDER=data
BUCKET_TRAINING_FOLDER=training

# name for the uploaded file inside of the bucket (we choose not to rename the file that we upload)
BUCKET_FILE_NAME_DATA=$(shell basename ${PATH_DATA_PROCESSED})
BUCKET_FILE_NAME_TARGET=$(shell basename ${PATH_TARGET_PROCESSED})

# project id - replace with your GCP project id
PROJECT_ID=meta-scanner-323307

# bucket name - replace with your GCP bucket name
BUCKET_NAME=kaggle-nfl-bowl-yards

# choose your region from https://cloud.google.com/storage/docs/locations#available_locations
REGION=europe-west1

gcp_submit_training:
	gcloud ai-platform jobs submit training ${JOB_NAME} \
  --job-dir gs://${BUCKET_NAME}/${BUCKET_TRAINING_FOLDER}  \
  --package-path ${PACKAGE_NAME} \
  --module-name ${PACKAGE_NAME}.${FILENAME} \
  --python-version=${PYTHON_VERSION} \
  --runtime-version=${RUNTIME_VERSION} \
  --region ${REGION} \
  --stream-logs


upload_data:
	gsutil cp $(PATH_DATA_PROCESSED) gs://$(BUCKET_NAME)/$(BUCKET_FOLDER)/$(BUCKET_FILE_NAME_DATA)
	gsutil cp $(PATH_TARGET_PROCESSED) gs://$(BUCKET_NAME)/$(BUCKET_FOLDER)/$(BUCKET_FILE_NAME_TARGET)

set_project:
	@gcloud config set project ${PROJECT_ID}

create_bucket:
	@gsutil mb -l ${REGION} -p ${PROJECT_ID} gs://${BUCKET_NAME}

run_locally:
	@python -m ${PACKAGE_NAME}.${FILENAME}

# ----------------------------------
#          INSTALL & TEST
# ----------------------------------
install_requirements:
	@pip install -r requirements.txt

check_code:
	@flake8 scripts/* kaggle_nflbowl_yards/*.py

black:
	@black scripts/* kaggle_nflbowl_yards/*.py

test:
	@coverage run -m pytest tests/*.py
	@coverage report -m --omit="${VIRTUAL_ENV}/lib/python*"

ftest:
	@Write me

clean:
	@rm -f */version.txt
	@rm -f .coverage
	@rm -fr */__pycache__ */*.pyc __pycache__
	@rm -fr build dist
	@rm -fr kaggle_nflbowl_yards-*.dist-info
	@rm -fr kaggle_nflbowl_yards.egg-info

install:
	@pip install . -U

all: clean install test black check_code

count_lines:
	@find ./ -name '*.py' -exec  wc -l {} \; | sort -n| awk \
        '{printf "%4s %s\n", $$1, $$2}{s+=$$0}END{print s}'
	@echo ''
	@find ./scripts -name '*-*' -exec  wc -l {} \; | sort -n| awk \
		        '{printf "%4s %s\n", $$1, $$2}{s+=$$0}END{print s}'
	@echo ''
	@find ./tests -name '*.py' -exec  wc -l {} \; | sort -n| awk \
        '{printf "%4s %s\n", $$1, $$2}{s+=$$0}END{print s}'
	@echo ''

# ----------------------------------
#      UPLOAD PACKAGE TO PYPI
# ----------------------------------
PYPI_USERNAME=<AUTHOR>
build:
	@python setup.py sdist bdist_wheel

pypi_test:
	@twine upload -r testpypi dist/* -u $(PYPI_USERNAME)

pypi:
	@twine upload dist/* -u $(PYPI_USERNAME)
