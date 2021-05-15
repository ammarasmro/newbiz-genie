all: conda-update pip-tools

# Arcane incantation to print all the other targets, from https://stackoverflow.com/a/26339924
help:
	@$(MAKE) -pRrq -f $(lastword $(MAKEFILE_LIST)) : 2>/dev/null | awk -v RS= -F: '/^# File/,/^# Finished Make data base/ {if ($$1 !~ "^[#.]") {print $$1}}' | sort | egrep -v -e '^[^[:alnum:]]' -e '^$@$$'

conda-update:
	conda env update --prune -f environment.yml

# Compile exact pip packages
pip-tools:
	pip install pip-tools
	pip-compile requirements/prod.in && pip-compile requirements/dev.in
	pip-sync requirements/prod.txt requirements/dev.txt
	python -m spacy download en_core_web_sm
	conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch-nightly -c nvidia

train-seq2seq:
	python training/run_experiment.py

run-web-app:
	streamlit run app/run.py

download-raw-dataset:
	kaggle datasets download -d peopledatalabssf/free-7-million-company-dataset -p data/kaggle_dataset --unzip
