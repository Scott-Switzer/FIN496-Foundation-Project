PYTHON := python3
PROJECT_ROOT := /Users/scottthomasswitzer/Desktop/FIN496FP/FIN496-Foundation-Project
ZIP_PATH := $(PROJECT_ROOT)/whitmore_taa_submission.zip
VERIFY_DIR := /tmp/whitmore_taa_verify

.PHONY: test pipeline zip verify-zip

test:
	$(PYTHON) -m pytest -q

pipeline:
	PYTHONPATH=. $(PYTHON) taa_project/main.py --start 2003-01-01 --end 2025-12-31 --folds 5

zip: pipeline
	cd $(PROJECT_ROOT) && zip -r $(ZIP_PATH) \
		taa_project \
		requirements.txt \
		.python-version \
		DECISIONS.md \
		TRIAL_LEDGER.csv \
		IPS.md \
		Guidelines.md \
		tasks.md \
		data/asset_data/whitmore_daily.csv \
		data/asset_data/data_key.csv \
		data/consolidated_csvs/fred/master/fred_data.csv \
		-x 'taa_project/**/__pycache__/*' \
		-x 'taa_project/**/*.pyc' \
		-x 'taa_project/outputs/*.csv' \
		-x 'taa_project/outputs/*.md' \
		-x 'taa_project/outputs/*.log' \
		-x 'taa_project/outputs/ablations/*' \
		-x 'taa_project/outputs/.mplconfig/*'

verify-zip: zip
	rm -rf $(VERIFY_DIR)
	mkdir -p $(VERIFY_DIR)
	unzip -q $(ZIP_PATH) -d $(VERIFY_DIR)
	cd $(VERIFY_DIR) && $(PYTHON) taa_project/main.py --start 2003-01-01 --end 2025-12-31 --folds 5 --no-timesfm
