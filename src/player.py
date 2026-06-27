"""
Player class for the LLM Mafia Game Competition.
"""
import re
import config
from typing import Optional
from openrouter import get_llm_response
from game_templates import (
    Role,
    GAME_RULES,
    CONFIRMATION_VOTE_EXPLANATIONS,
    PROMPT_TEMPLATES,
    CONFIRMATION_VOTE_TEMPLATES,
    THINKING_TAGS,
    ACTION_PATTERNS,
    VOTE_PATTERNS,
    CONFIRMATION_VOTE_PATTERNS,
)
from trust_graph import TrustGraph
from prompt_builder import PromptBuilder
from phrase_graph import PhraseGraph
from big_five import BigFiveProfile


class Player:
    """Represents an LLM player in the Mafia game."""

    def __init__(self, model_name, player_name, role, language=None, use_graph=False, use_phrase_graph=False, use_big_five=False, game=None):
        """
        Initialize a player.

        Args:
            model_name (str): The name of the LLM model (hidden from other players, used only for API calls).
            player_name (str): The unique visible name of the player in the game (used for all identification).
            role (Role): The role of the player in the game.
            language (str, optional): The language for the player. Defaults to English.
            use_graph (bool, optional): Whether this player uses graph-based reasoning.
            use_phrase_graph (bool, optional): Whether this player uses per-phrase relationship graphs.
            use_big_five (bool, optional): Whether this player performs Big Five assessments.
            game (MafiaGame, optional): Reference to the game instance.
        """
        self.model_name = model_name      # Hidden: only used for LLM API calls
        self.player_name = player_name    # Visible: used for all in-game identification
        self.role = role
        self.alive = True
        self.use_graph = use_graph
        self.use_phrase_graph = use_phrase_graph  # per-phrase graph activation
        self.use_big_five = use_big_five          # Big Five assessment activation
        self.bigfive_profile: Optional[BigFiveProfile] = None
        self.bigfive_assessments: list = []       # accumulated per-observer estimates
        self.trust_graph = None           # TrustGraph instance (replaces self.graph)
        self.protected = False            # Whether the player is protected by the doctor
        self.language = language if language else "English"
        self.game = game
        self.prompt_builder = PromptBuilder()
        self.graph_sequence = []          # Stores per-phrase relationship graphs

        # Store the most recent private thoughts extracted from the LLM response
        self.last_think = ""

    def __str__(self):
        """Return a string representation of the player."""
        return f"{self.player_name} ({self.role.value}) [Model: {self.model_name}] ({self.use_graph=})"

    # ------------------------------------------------------------------
    # Graph delegation methods
    # ------------------------------------------------------------------

    def init_graph(self, all_players):
        """
        Initialize subjective directed trust graph for this player.
        Delegates to TrustGraph.
        """
        if not self.use_graph:
            return None

        self.trust_graph = TrustGraph(self.player_name, self.role, all_players)
        return self.trust_graph.graph

    def graph_to_prompt(self, all_players):
        """
        Convert subjective graph to compact text description for LLM prompt.
        Delegates to TrustGraph.
        """
        if not self.use_graph or self.trust_graph is None:
            return ""

        return self.trust_graph.to_prompt(all_players)

    def update_graph(self, all_players, current_round):
        """
        Update the subjective trust graph based on the last round's discussion.
        Delegates to TrustGraph.

        Args:
            all_players (list): List of all players in the game.
            current_round (int): Current round number.
        """
        if not self.use_graph or self.trust_graph is None:
            return

        last_round_history = self.discussion_history_last_round_without_thinking()

        self.trust_graph.update(
            all_players=all_players,
            current_round=current_round,
            discussion_history=last_round_history,
            model_name=self.model_name,
        )

    # ------------------------------------------------------------------
    # Per-phrase graph sequence methods
    # ------------------------------------------------------------------

    def generate_per_phrase_graph(self, message, speaker_name, alive_players, current_round, phase):
        """
        Generate a per-phrase relationship graph for the given message and
        append it to the graph_sequence.

        The player's own LLM model (self.model_name) is used to evaluate the
        attitude weights toward each mentioned player.

        Args:
            message (str): The message spoken by the player.
            speaker_name (str): player-speaker
            alive_players (list): List of currently alive Player objects.
            current_round (int): Current round number.
            phase (str): Current game phase string.
        """
        graph_data = PhraseGraph.build_graph(
            message=message,
            speaker=speaker_name,
            alive_players=alive_players,
            current_round=current_round,
            phase=phase,
            model_name=self.model_name,  # Use this player's own LLM to assign weights
        )
        self.graph_sequence.append(graph_data)

    def graph_sequence_to_dict(self):
        """
        Return the graph sequence as a JSON-serializable list.

        Returns:
            list: The list of graph dictionaries.
        """
        return self.graph_sequence

    def graph_sequence_from_dict(self, data):
        """
        Restore the graph sequence from a list of dictionaries.

        Args:
            data (list): List of graph dictionaries to restore.
        """
        self.graph_sequence = data

    # ------------------------------------------------------------------
    # Discussion history helpers
    # ------------------------------------------------------------------

    def discussion_history_without_thinking(self):
        return self.game.discussion_history_without_thinking()

    def discussion_history_last_round_without_thinking(self):
        return self.game.discussion_history_last_round_without_thinking()

    # ------------------------------------------------------------------
    # Prompt generation
    # ------------------------------------------------------------------

    def _find_target_player(self, target_name, all_players, exclude_mafia=False):
        """
        Find a target player by player_name.

        Args:
            target_name (str): The player_name of the target player.
            all_players (list): List of all players in the game.
            exclude_mafia (bool, optional): Whether to exclude Mafia members from targets.

        Returns:
            Player or None: The target player if found, None otherwise.
        """
        for player in all_players:
            if not player.alive:
                continue

            if exclude_mafia and player.role == Role.MAFIA:
                continue

            # Match against player_name only (not model_name)
            if target_name.lower() in player.player_name.lower():
                return player

        return None

    def generate_prompt(
        self, game_state, all_players, mafia_members=None, discussion_history=None
    ):
        """
        Generate a prompt for the player based on their role.
        Delegates to PromptBuilder.

        Args:
            game_state (dict): The current state of the game.
            all_players (list): List of all players in the game.
            mafia_members (list, optional): List of mafia members (only for Mafia role).
            discussion_history (str, optional): History of previous discussions.

        Returns:
            str: The prompt for the player.
        """
        return self.prompt_builder.build_discussion_prompt(
            player=self,
            game_state=game_state,
            all_players=all_players,
            mafia_members=mafia_members,
            discussion_history=discussion_history,
        )

    # ------------------------------------------------------------------
    # LLM interaction
    # ------------------------------------------------------------------

    def get_response(self, prompt):
        """
        Get a response from the LLM model using OpenRouter API.
        Uses model_name for the API call (hidden from other players).

        The response is split into private thoughts (inside <think> tags)
        and public speech (the rest).  Both parts are stored; callers can
        access self.last_think to retrieve the private part.

        Args:
            prompt (str): The prompt to send to the model.

        Returns:
            str: The public part of the response (thoughts removed).
        """
        print(f"\n\n{len(prompt)=} {prompt=}\n\n")
        raw_response = get_llm_response(self.model_name, prompt)

        # Extract private thoughts from <think>...</think> blocks
        think_pattern = re.compile(
            r"<[tT][hH][iI][nN][kK]>(.*?)</[tT][hH][iI][nN][kK]>",
            re.DOTALL,
        )
        think_matches = think_pattern.findall(raw_response)
        self.last_think = "\n".join(think_matches).strip() if think_matches else ""

        # Remove all <think>...</think> blocks to obtain the public message
        public_response = re.sub(
            r"<[tT][hH][iI][nN][kK]>.*?</[tT][hH][iI][nN][kK]>",
            "",
            raw_response,
            flags=re.DOTALL,
        )

        # Clean up excessive whitespace
        public_response = re.sub(r"\n\s*\n", "\n\n", public_response)
        public_response = public_response.strip()

        return public_response

    # ------------------------------------------------------------------
    # Response parsing
    # ------------------------------------------------------------------

    def parse_night_action(self, response, all_players):
        """
        Parse the night action from the player's response.
        Targets are identified by player_name.

        Args:
            response (str): The response from the player (already cleaned of thinking tags).
            all_players (list): List of all players in the game.

        Returns:
            tuple: (action_type, target_player) or (None, None) if no valid action.
        """
        if self.role == Role.MAFIA:
            pattern = ACTION_PATTERNS.get(self.language, ACTION_PATTERNS["English"])[
                Role.MAFIA
            ]
            match = re.search(pattern, response, re.IGNORECASE)

            if match:
                target_name = match.group(1).strip()
                # Find target by player_name, excluding Mafia members
                target_player = self._find_target_player(
                    target_name, all_players, exclude_mafia=True
                )
                if target_player:
                    return "kill", target_player
            return None, None

        elif self.role == Role.DOCTOR:
            pattern = ACTION_PATTERNS.get(self.language, ACTION_PATTERNS["English"])[
                Role.DOCTOR
            ]
            match = re.search(pattern, response, re.IGNORECASE)

            if match:
                target_name = match.group(1).strip()
                # Find target by player_name
                target_player = self._find_target_player(target_name, all_players)
                if target_player:
                    return "protect", target_player
            return None, None
        else:
            # Villagers don't have night actions
            return None, None

    def parse_day_vote(self, response, all_players):
        """
        Parse the day vote from the player's response.
        Vote target is identified by player_name.

        Args:
            response (str): The response from the player (already cleaned of thinking tags).
            all_players (list): List of all players in the game.

        Returns:
            Player or None: The player being voted for, or None if no valid vote.
        """
        pattern = VOTE_PATTERNS.get(self.language, VOTE_PATTERNS["English"])
        match = re.search(pattern, response, re.IGNORECASE)

        if match:
            target_name = match.group(1).strip()
            # Find target by player_name
            return self._find_target_player(target_name, all_players)
        return None

    def get_confirmation_vote(self, game_state, all_players, discussion_history):
        """
        Get a confirmation vote from the player on whether to eliminate another player.
        The player to eliminate is identified by player_name.
        Delegates prompt building to PromptBuilder.

        Args:
            game_state (dict): The current state of the game, including who is up for elimination.
            all_players (list): List of all players.
            discussion_history (str): Discussion history string.

        Returns:
            str: "agree" or "disagree" indicating the player's vote
        """
        prompt = self.prompt_builder.build_confirmation_vote_prompt(
            player=self,
            game_state=game_state,
            all_players=all_players,
            discussion_history=discussion_history,
        )

        # Ensure game configuration information appears in the prompt
        if self.game:
            total_mafia = sum(1 for p in self.game.players if p.role == Role.MAFIA)
            total_doctor = sum(1 for p in self.game.players if p.role == Role.DOCTOR)
            total_villager = sum(1 for p in self.game.players if p.role == Role.VILLAGER)
            doctor_word = "Doctors" if total_doctor != 1 else "Doctor"
            config_line = (
                f"Game configuration: {total_villager} Villagers, "
                f"{total_mafia} Mafia, and {total_doctor} {doctor_word}."
            )
            if config_line not in prompt:
                prompt = config_line + "\n" + prompt

        # Prepend role information if not already present
        role_phrase = f"You are a {self.role.value}. "
        if not prompt.startswith(role_phrase):
            prompt = role_phrase + prompt

        response = self.get_response(prompt)

        # Log private thoughts, if any
        if self.last_think:
            self.game.logger.player_thoughts(
                self.model_name,
                self.role.value,
                self.last_think,
                player_name=self.player_name,
            )

        # Log the public response
        self.game.logger.player_response(
            self.model_name,
            self.role.value,
            response,
            player_name=self.player_name,
        )

        # Parse the response for agree/disagree based on language
        language = (
            self.language if self.language in CONFIRMATION_VOTE_PATTERNS else "English"
        )
        if re.search(CONFIRMATION_VOTE_PATTERNS[language]["agree"], response.lower()):
            return "agree"
        else:
            return "disagree"
