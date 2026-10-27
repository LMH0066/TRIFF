# Develop Guidelines
## environment prepare
```bash
conda create -n TRIFF python=3.10
conda activate TRIFF
pip install poetry
poetry install
```
## pre commit
```bash
# Format python code
poe format
# Run test code
poe test
```
