## Project Setup

### Python Environment
Note these steps assume you have a python/pip alias. If you don't, you may want to install `pyenv` to manage your python environment: 
```bash
brew install pyenv
```

1. Create the virtual environment: 
```bash 
python -m venv .venv
```
2. Activate the environment:
```bash
source .venv/bin/activate
```
3. Install the required packages: 
```bash
pip install -r requirements.txt
```

### Configuration
1. Create a `.env` file and add your OpenAI API key
