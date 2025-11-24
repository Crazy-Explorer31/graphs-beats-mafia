# LLM Mafia Game

Follow this steps to run:
* Pull to your ollama models and write them into `src/config.py --- OLLAMA_MODELS`
* Run `ollama serve`
* Pull dependencies with `uv sync`
* Run `uv run src/simulate.py  --ollama`
* See logs in `logs/` dir
