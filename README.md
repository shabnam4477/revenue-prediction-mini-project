# Revenue Prediction Mini-project
This project is a python script using machine learning models to predict revenue for open banking data

## Reports
Find the full observation report in `reports/report.md`
## Logs
Find the prediction result log in `debug.log`
## Notebooks
Find the data analyse steps and results in `notebooks/Data_Analyse.ipynb`

# Configure the project
## Pre-requiremets
- `python >=3.12`
- `uv`
- `make`

## Requirements
**Create and activate virtual environment**
```
uv venv --python 3.12
source .venv/bin/activate.fish
```

**Install requirements**

```
uv sync
```
or
```
uv pip install -r requirements.txt
```
## Run the script
```
uv run python src/main.py
```