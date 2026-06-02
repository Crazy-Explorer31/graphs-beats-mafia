"""
Game logic for the LLM Mafia Game Competition.
"""

import random
import uuid
from player import Player
from game_templates import Role
import config
from logger import GameLogger, Color
from game_state import GameStateManager, DiscussionHistory
from game_orchestrator import GameOrchestrator
from game_phases import NightExecutor, DayExecutor


class MafiaGame:
    """Represents a Mafia game with LLM players."""

    def __init__(self, models=None, language=None):
        """
        Initialize a Mafia game.

        Args:
            models (list, optional): List of model names to use as players.
            language (str, optional): Language for game prompts and interactions. Defaults to config.LANGUAGE.
        """
        self.game_id = str(uuid.uuid4())
        self.round_number = 0
        self.phase = "setup"

        # state manager
        self.language = language if language is not None else config.LANGUAGE
        self.state = GameStateManager(self.language)
        self.disc = DiscussionHistory()

        # expose shared lists/dicts from state for seamless migration
        self.players = self.state.players
        self.mafia_players = self.state.mafia_players
        self.doctor_player = self.state.doctor_player   # will be set later
        self.villager_players = self.state.villager_players
        self.rounds_data = self.state.rounds_data
        self.current_round_data = self.state.current_round_data

        self.discussion_history = ""
        self.discussion_history_last_round = ""

        self.unique_models = config.UNIQUE_MODELS

        # Use provided models or default from config
        self.models = models if models else config.MODELS

        # Set random seed if specified
        if config.RANDOM_SEED is not None:
            random.seed(config.RANDOM_SEED)

        # Initialize logger
        self.logger = GameLogger()

        # Initialize orchestrator (must be last after state and logger)
        self.orchestrator = GameOrchestrator(self)

        # Initialize phase executors
        self.night_executor = NightExecutor(self)
        self.day_executor = DayExecutor(self)

    def setup_game(self):
        """
        Set up the game by assigning roles to players.

        Returns:
            bool: True if setup successful, False otherwise.
        """
        # Check if we have enough models
        if len(self.models) < config.PLAYERS_PER_GAME and self.unique_models:
            self.logger.error(
                f"Not enough models. Need {config.PLAYERS_PER_GAME}, but only have {len(self.models)}."
            )
            return False

        # Log game start
        self.logger.game_start(1, self.game_id, self.language)

        # Randomly select models for this game
        if self.unique_models:
            selected_models = random.sample(self.models, config.PLAYERS_PER_GAME)
        else:
            selected_models = random.choices(self.models, k=config.PLAYERS_PER_GAME)

        # Assign roles
        roles = []

        # Add Mafia roles
        for _ in range(config.MAFIA_COUNT):
            roles.append(Role.MAFIA)

        # Add Doctor roles
        for _ in range(config.DOCTOR_COUNT):
            roles.append(Role.DOCTOR)

        # Add Villager roles
        villager_count = (
            config.PLAYERS_PER_GAME - config.MAFIA_COUNT - config.DOCTOR_COUNT
        )
        for _ in range(villager_count):
            roles.append(Role.VILLAGER)

        # Shuffle roles
        random.shuffle(roles)

        # Create players
        self.logger.header("PLAYER SETUP", Color.CYAN)
        for i, model_name in enumerate(selected_models):
            # Generate a unique player name (not based on model name)
            used_names = [p.player_name for p in self.players]
            available_names = [name for name in player_names if name not in used_names]

            # If we somehow run out of names, use a numbered fallback
            if not available_names:
                player_name = f"Player_{i+1}"
            else:
                player_name = random.choice(available_names)

            # use_graph = False
            # if roles[i] == Role.VILLAGER or roles[i] == Role.DOCTOR:
            use_graph = True

            # Create player with both model_name (hidden) and player_name (visible)
            player = Player(model_name, player_name, roles[i], language=self.language, game=self, use_graph=use_graph)
            self.players.append(player)

            # Add to role-specific lists
            if player.role == Role.MAFIA:
                self.mafia_players.append(player)
            elif player.role == Role.DOCTOR:
                self.doctor_player = player
            else:  # Role.VILLAGER
                self.villager_players.append(player)

            # Log player setup using player_name as the visible identifier
            self.logger.player_setup(
                player.player_name, player.role.value, player.player_name
            )

        # Set phase to night
        self.phase = "night"
        self.round_number = 1
        self.current_round_data = {
            "round_number": self.round_number,
            "messages": [],
            "actions": {},
            "eliminations": [],
            "eliminated_by_vote": [],  # Reset for the new round
            "targeted_by_mafia": [],   # Reset for the new round
            "protected_by_doctor": [], # Reset for the new round
            "outcome": "",
        }

        # Sync state after setup
        self.state.players = self.players
        self.state.mafia_players = self.mafia_players
        self.state.doctor_player = self.doctor_player
        self.state.villager_players = self.villager_players
        self.state.round_number = self.round_number
        self.state.phase = self.phase
        self.state.current_round_data = self.current_round_data

        return True

    # ---- State delegation methods ----

    def get_game_state(self):
        """Return formatted game state, delegating to GameStateManager."""
        self.state.round_number = self.round_number
        self.state.phase = self.phase
        self.state.current_round_data = self.current_round_data
        return self.state.get_game_state()

    def get_alive_players(self):
        """Return list of alive players, delegating to GameStateManager."""
        return self.state.get_alive_players()

    def check_game_over(self):
        """Check if game is over, delegating to GameStateManager."""
        self.state.round_number = self.round_number
        return self.state.check_game_over()

    def discussion_history_without_thinking(self):
        """Return discussion history without think tags, delegated to DiscussionHistory."""
        return self.disc.without_thinking(self.discussion_history)

    def discussion_history_last_round_without_thinking(self):
        """Return last round history without think tags, delegated to DiscussionHistory."""
        return self.disc.without_thinking(self.discussion_history_last_round)

    # ---- Phase execution ----

    def execute_night_phase(self):
        """
        Execute the night phase of the game.

        Returns:
            list: List of eliminated players.
        """
        return self.night_executor.execute()

    def execute_day_phase(self):
        """
        Execute the day phase of the game.

        Returns:
            list: List of eliminated players.
        """
        return self.day_executor.execute()

    # ---- Orchestration delegation ----

    def init_all_graphs(self):
        """Initialize graphs for all players, delegated to GameOrchestrator."""
        self.orchestrator.init_all_graphs()

    def update_all_graphs(self, current_round):
        """Update graphs for all players, delegated to GameOrchestrator."""
        self.orchestrator.update_all_graphs(current_round)

    def run_game(self):
        """Run the Mafia game until completion, delegated to GameOrchestrator."""
        return self.orchestrator.run_game()

    def generate_critic_review(self, winner):
        """Generate a game critic review, delegated to GameOrchestrator."""
        return self.orchestrator.generate_critic_review(winner)


player_names = [
    "Alex",
    "Bailey",
    "Casey",
    "Dana",
    "Ellis",
    "Finley",
    "Gray",
    "Harper",
    "Indigo",
    "Jordan",
    "Kennedy",
    "Logan",
    "Morgan",
    "Nico",
    "Parker",
    "Quinn",
    "Riley",
    "Sage",
    "Taylor",
    "Avery",
    "Blake",
    "Cameron",
    "Drew",
    "Emerson",
    "Frankie",
    "Hayden",
    "Jamie",
    "Kai",
    "Leighton",
    "Marley",
    "Noel",
    "Oakley",
    "Peyton",
    "Reese",
    "Skyler",
    "Tatum",
    "Val",
    "Winter",
    "Zion",
]
