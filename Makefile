install:
	python -m pip install -r requirements.txt

train:
	PYTHONPATH=src python train.py

report:
	PYTHONPATH=src python make_report.py

test:
	PYTHONPATH=src pytest -q
