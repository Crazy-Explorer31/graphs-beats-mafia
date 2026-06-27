"""
Configuration settings for the LLM Mafia Game Competition.
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# RouterAI API settings (replaces OpenRouter)
ROUTERAI_API_KEY = os.getenv("ROUTERAI_API_KEY")
if ROUTERAI_API_KEY is None:
    raise ValueError("`ROUTERAI_API_KEY` environment variable is not set")

ROUTERAI_API_URL = os.getenv(
    "ROUTERAI_API_URL",
    "https://routerai.ru/api/v1/chat/completions",
)

# OpenRouter API settings (deprecated, kept for reference)
# OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
# OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"

# Ollama API settings
OLLAMA_API_URL = os.getenv("OLLAMA_API_URL", "http://localhost:11434/api/generate")

# ---- SWITCH TO ROUTERAI ONLY ----
# Setting OLLAMA_MODELS to an empty list ensures that all player models
# are handled by RouterAI (or any non‑Ollama API you configure).
OLLAMA_MODELS = []  # formerly: ["llama3.2:3b"] * 6

# ROUTERAI_MODELS works exactly like the old OLLAMA_MODELS list:
# each element corresponds to the model that a particular player will use.
# Here we assign all six players to deepseek/deepseek-v4-flash.
ROUTERAI_MODELS = ["deepseek/deepseek-v4-flash"] * 6

# Firebase settings
FIREBASE_CREDENTIALS_PATH = (
    "firebase_credentials.json"  # Path to your Firebase credentials file
)

# Game settings

CLAUDE_SONNET_4 = "anthropic/claude-sonnet-4"
MODELS = [
    # DeepSeek (via RouterAI)
    "deepseek/deepseek-v4-flash",
    # Google
    "google/gemini-flash-1.5",
    "google/gemini-flash-1.5-8b",
    "google/gemini-2.0-flash-lite-001",
    "google/gemini-2.0-flash-001",
    "google/gemini-2.5-flash",
    "google/gemini-2.5-pro",
    # Meta
    "meta-llama/llama-4-maverick",
    "meta-llama/llama-3.3-70b-instruct",
    "meta-llama/llama-3.1-70b-instruct",
    # DeepSeek
    "deepseek/deepseek-chat",
    "deepseek/deepseek-chat-v3-0324",
    "deepseek/deepseek-r1-distill-llama-70b",
    # OpenAI
    "openai/gpt-4o-mini",
    "openai/gpt-4o",
    "openai/gpt-4.1-mini",
    "openai/gpt-4.1",
    # Anthropic
    "anthropic/claude-3.5-sonnet",
    "anthropic/claude-3.7-sonnet",
    "anthropic/claude-3.7-sonnet:thinking",
    "anthropic/claude-sonnet-4",
    # Qwen
    "qwen/qwen3-coder",
    # Other
    "microsoft/wizardlm-2-8x22b",
    "mistralai/mistral-small-24b-instruct-2501",
    "nousresearch/hermes-3-llama-3.1-405b",
    "minimax/minimax-01",
    "gryphe/mythomax-l2-13b",
    "moonshotai/kimi-k2",
]

FREE_MODELS = [
    "deepseek/deepseek-r1:free",
    "google/gemini-2.0-pro-exp-02-05:free",
    "google/gemini-2.0-flash-exp:free",
    "google/gemini-2.0-flash-lite-preview-02-05:free",
    "meta-llama/llama-3.3-70b-instruct:free",
    "deepseek/deepseek-r1-distill-llama-70b:free",
    "google/gemini-exp-1206:free",
    "qwen/qwen2.5-vl-72b-instruct:free",
    "cognitivecomputations/dolphin3.0-r1-mistral-24b:free",
    "nvidia/llama-3.1-nemotron-70b-instruct:free",
]

# Game configuration
NUM_GAMES = int(os.getenv("NUM_GAMES", 1))  # Number of games to simulate
PLAYERS_PER_GAME = int(
    os.getenv("PLAYERS_PER_GAME", 6)
)  # Number of players in each game
MAFIA_COUNT = int(os.getenv("MAFIA_COUNT", 2))  # Number of Mafia players
DOCTOR_COUNT = int(os.getenv("DOCTOR_COUNT", 1))  # Number of Doctor players
# Villagers will be: PLAYERS_PER_GAME - MAFIA_COUNT - DOCTOR_COUNT

# Game type
GAME_TYPE = "Classic Mafia"  # Type of Mafia game to run

# Language setting
LANGUAGE = os.getenv(
    "GAME_LANGUAGE", "English"
)  # Language for game prompts and interactions (supported: English, Spanish, French, Korean)

# Maximum number of rounds before declaring a draw
MAX_ROUNDS = int(os.getenv("MAX_ROUNDS", 20))

# Timeout for API calls (in seconds)
API_TIMEOUT = int(os.getenv("API_TIMEOUT", 60))

# Maximum output tokens for LLM responses
MAX_OUTPUT_TOKENS = int(os.getenv("MAX_OUTPUT_TOKENS", 8192))

# Model-specific configurations
MODEL_CONFIGS = {
    "deepseek/deepseek-v4-flash": {
        "timeout": 90,      # tune if needed
    },
    "deepseek/deepseek-r1": {
        "timeout": 90,  # Longer timeout for DeepSeek-R1
    },
    "deepseek/deepseek-r1:free": {
        "timeout": 90,
    },
    "deepseek/deepseek-r1-distill-llama-70b": {
        "timeout": 90,
    },
    "deepseek/deepseek-r1-distill-llama-70b:free": {
        "timeout": 90,
    },
    "deepseek/deepseek-chat": {
        "timeout": 90,
    },
}

# Random seed for reproducibility (set to None for random behavior)
RANDOM_SEED = os.getenv("RANDOM_SEED")
if RANDOM_SEED is not None:
    RANDOM_SEED = int(RANDOM_SEED)

# For testing with a single RouterAI model, allow repeated use of the same model.
# Set this to False so that multiple players can all use deepseek-v4-flash.
UNIQUE_MODELS = False   # was: os.getenv("UNIQUE_MODELS", "true") == "true"
