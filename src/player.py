"""
Player class for the LLM Mafia Game Competition.
"""
import networkx as nx
import numpy as np
import random
import re
import config
from openrouter import get_llm_response
from game_templates import (
    Role,
    GAME_RULES,
    CONFIRMATION_VOTE_EXPLANATIONS,
    PROMPT_TEMPLATES,
    PROMPT_TEMPLATES_HIDDEN,
    CONFIRMATION_VOTE_TEMPLATES,
    THINKING_TAGS,
    ACTION_PATTERNS,
    VOTE_PATTERNS,
    CONFIRMATION_VOTE_PATTERNS,
)


class Player:
    """Represents an LLM player in the Mafia game."""

    def __init__(self, model_name, player_name, role, language=None, use_graph=False):
        """
        Initialize a player.

        Args:
            model_name (str): The name of the LLM model to use for this player (hidden from other players).
            player_name (str): The visible name of the player in the game.
            role (Role): The role of the player in the game.
            language (str, optional): The language for the player. Defaults to English.
        """
        self.model_name = model_name
        self.player_name = player_name
        self.role = role
        self.alive = True
        self.use_graph = use_graph
        self.graph = None
        self.protected = False  # Whether the player is protected by the doctor
        self.language = language if language else "English"

    def __str__(self):
        """Return a string representation of the player."""
        return f"{self.player_name} ({self.role.value}) [Model: {self.model_name}] ({self.use_graph=})"

    def init_graph(self, all_players) -> nx.DiGraph:
        """
        Initialize subjective directed trust graph for this player.
        Vertices: all players with their role probabilities.
        Edges: trust values between players (-1 to 1).
        Player knows their own role, so adjusts probabilities for others.
        Mafia players know each other at game start.
        """
        if not self.use_graph:
            return None

        G = nx.DiGraph()

        # Get total counts from config
        total_players = config.PLAYERS_PER_GAME
        mafia_count = config.MAFIA_COUNT
        doctor_count = config.DOCTOR_COUNT
        villager_count = total_players - mafia_count - doctor_count

        # Identify all mafia players
        all_mafia_players = [p.player_name for p in all_players if p.role.value == "Mafia"]

        # Adjust counts: remove self from appropriate category
        remaining_mafia = mafia_count
        remaining_doctor = doctor_count
        remaining_villager = villager_count

        if self.role.value == "Mafia":
            remaining_mafia -= 1
        elif self.role.value == "Doctor":
            remaining_doctor -= 1
        elif self.role.value == "Villager":
            remaining_villager -= 1

        # Calculate probabilities for other players (excluding self)
        other_players_count = total_players - 1
        if other_players_count > 0:
            other_mafia_prob = remaining_mafia / other_players_count
            other_doctor_prob = remaining_doctor / other_players_count
            other_villager_prob = remaining_villager / other_players_count
        else:
            other_mafia_prob = other_doctor_prob = other_villager_prob = 0

        # Add all players as nodes
        for player in all_players:
            if player.player_name == self.player_name:
                # Self: know exact role
                role_probs = {
                    "Mafia": 1.0 if self.role.value == "Mafia" else 0.0,
                    "Villager": 1.0 if self.role.value == "Villager" else 0.0,
                    "Doctor": 1.0 if self.role.value == "Doctor" else 0.0
                }
                self_trust = 1.0
            else:
                # Mafia players know each other
                if self.role.value == "Mafia" and player.player_name in all_mafia_players:
                    # This is another mafia - we know their exact role
                    role_probs = {"Mafia": 1.0, "Villager": 0.0, "Doctor": 0.0}
                else:
                    # Others: use calculated probabilities
                    role_probs = {
                        "Mafia": other_mafia_prob,
                        "Villager": other_villager_prob,
                        "Doctor": other_doctor_prob
                    }
                # Others trust themselves neutrally from our perspective
                self_trust = 0.0

            G.add_node(
                player.player_name,
                role_probabilities=role_probs,
                alive=True,
                is_self=(player.player_name == self.player_name),
                actual_role=player.role.value if player.player_name == self.player_name else None
            )

            # Self-loop for self-trust
            G.add_edge(
                player.player_name,
                player.player_name,
                trust=self_trust,
                evidence=[],
                last_updated=0
            )

        # Initialize trust edges between all players
        for player1 in all_players:
            for player2 in all_players:
                if player1.player_name != player2.player_name:
                    # Mafia trust each other more at start
                    initial_trust = 0.0
                    if (self.role.value == "Mafia" and
                            player1.player_name in all_mafia_players and
                            player2.player_name in all_mafia_players):
                        initial_trust = 0.5  # Moderate trust between mafia members

                    G.add_edge(
                        player1.player_name,
                        player2.player_name,
                        trust=initial_trust,
                        evidence=[],
                        last_updated=0
                    )

        self.graph = G
        return G

    def graph_to_prompt(self, all_players):
        """
        Convert subjective graph to compact text description for LLM prompt.
        """
        if not self.use_graph or self.graph is None:
            return ""

        alive_players = [p for p in all_players if p.alive]

        lines = ["YOUR TRUST GRAPH:"]

        # My trust in others
        my_trust = []
        for player in alive_players:
            if player.player_name == self.player_name:
                continue

            # Get trust from me to them
            trust = self.graph[self.player_name][player.player_name]['trust']

            # Get role probabilities
            probs = self.graph.nodes[player.player_name]['role_probabilities']
            mafia_prob = probs.get("Mafia", 0.0)

            # Format trust level
            if trust > 0.5:
                trust_level = "highly trust"
            elif trust > 0.2:
                trust_level = "trust"
            elif trust < -0.5:
                trust_level = "highly distrust"
            elif trust < -0.2:
                trust_level = "distrust"
            else:
                trust_level = "neutral"

            my_trust.append(f"  - {player.player_name}: {trust_level}, Mafia prob: {mafia_prob:.0%}")

        if my_trust:
            lines.append("My trust in others:")
            lines.extend(my_trust)

        # Summary: most trusted and suspicious
        trusted = []
        suspicious = []

        for player in alive_players:
            if player.player_name == self.player_name:
                continue

            trust = self.graph[self.player_name][player.player_name]['trust']
            mafia_prob = self.graph.nodes[player.player_name]['role_probabilities'].get("Mafia", 0.0)

            if trust > 0.3:
                trusted.append((player.player_name, trust))
            if mafia_prob > 0.4:
                suspicious.append((player.player_name, mafia_prob))

        # Take top 2-3
        trusted.sort(key=lambda x: x[1], reverse=True)
        suspicious.sort(key=lambda x: x[1], reverse=True)

        if trusted:
            top_trusted = ", ".join([f"{name}" for name, _ in trusted[:2]])
            lines.append(f"\nMost trusted: {top_trusted}")

        if suspicious:
            top_suspicious = ", ".join([f"{name}" for name, _ in suspicious[:2]])
            lines.append(f"Most suspicious: {top_suspicious}")

        # Key mutual relationships
        mutual_relations = []
        for player1 in alive_players:
            for player2 in alive_players:
                if (player1.player_name == self.player_name or
                        player2.player_name == self.player_name or
                        player1.player_name == player2.player_name):
                    continue

                trust1 = self.graph[player1.player_name][player2.player_name]['trust']
                trust2 = self.graph[player2.player_name][player1.player_name]['trust']

                # Strong mutual trust or distrust
                if trust1 > 0.4 and trust2 > 0.4:
                    mutual_relations.append(f"  - {player1.player_name} ↔ {player2.player_name} trust each other")
                elif trust1 < -0.3 and trust2 < -0.3:
                    mutual_relations.append(f"  - {player1.player_name} ↔ {player2.player_name} distrust each other")

        if mutual_relations:
            lines.append("\nKey relationships:")
            lines.extend(mutual_relations[:3])

        return "\n".join(lines)

    def _find_target_player(self, target_name, all_players, exclude_mafia=False):
        """
        Find a target player by name.

        Args:
            target_name (str): The name of the target player.
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

            if target_name.lower() in player.player_name.lower():
                return player

        return None

    def generate_prompt(
        self, game_state, all_players, mafia_members=None, discussion_history=None
    ):
        """
        Generate a prompt for the player based on their role.

        Args:
            game_state (dict): The current state of the game.
            all_players (list): List of all players in the game.
            mafia_members (list, optional): List of mafia members (only for Mafia role).
            discussion_history (str, optional): History of previous discussions.
                Note: This should only contain day phase messages, night messages are filtered out.

        Returns:
            str: The prompt for the player.
        """
        if discussion_history is None:
            discussion_history = ""

        # Get list of player names (using visible player names)
        player_names = [p.player_name for p in all_players if p.alive]

        # Make sure we're only using player_name (not model_name) for other players
        # This ensures players only know each other by their player names
        player_info = [{"name": p.player_name, "alive": p.alive} for p in all_players]

        # Get the appropriate language, defaulting to English if not supported
        language = self.language if self.language in GAME_RULES else "English"

        # Get game rules for the player's language
        game_rules = GAME_RULES[language]

        if self.role == Role.MAFIA:
            # For Mafia members (using visible player names)
            mafia_names = [
                p.player_name for p in mafia_members if p != self and p.alive
            ]
            mafia_list = f"{', '.join(mafia_names) if mafia_names else 'None (you are the only Mafia left)'}"
            if language == "Spanish":
                mafia_list = f"{', '.join(mafia_names) if mafia_names else 'Ninguno (eres el único miembro de la Mafia que queda)'}"
            elif language == "French":
                mafia_list = f"{', '.join(mafia_names) if mafia_names else 'Aucun (vous êtes le seul membre de la Mafia restant)'}"
            elif language == "Korean":
                mafia_list = f"{', '.join(mafia_names) if mafia_names else '없음 (당신이 유일하게 남은 마피아입니다)'}"

            prompt = PROMPT_TEMPLATES[language][Role.MAFIA].format(
                model_name=self.player_name,  # Use player_name in prompts
                game_rules=game_rules,
                mafia_members=mafia_list,
                player_names=", ".join(player_names),
                game_state=game_state,
                thinking_tag=THINKING_TAGS[language],
                discussion_history=discussion_history,
            )
        elif self.role == Role.DOCTOR:
            # For Doctor
            prompt = PROMPT_TEMPLATES[language][Role.DOCTOR].format(
                model_name=self.player_name,  # Use player_name in prompts
                game_rules=game_rules,
                player_names=", ".join(player_names),
                game_state=game_state,
                thinking_tag=THINKING_TAGS[language],
                discussion_history=discussion_history,
            )
        else:  # Role.VILLAGER
            # For Villagers
            prompt = PROMPT_TEMPLATES[language][Role.VILLAGER].format(
                model_name=self.player_name,  # Use player_name in prompts
                game_rules=game_rules,
                player_names=", ".join(player_names),
                game_state=game_state,
                thinking_tag=THINKING_TAGS[language],
                discussion_history=discussion_history,
            )

        return prompt
    
    def generate_prompt_hidden(
        self, game_state, all_players, mafia_members=None, discussion_history=None, question=None
    ):
        """
        Generate a prompt for the player based on their role.

        Args:
            game_state (dict): The current state of the game.
            all_players (list): List of all players in the game.
            mafia_members (list, optional): List of mafia members (only for Mafia role).
            discussion_history (str, optional): History of previous discussions.
                Note: This should only contain day phase messages, night messages are filtered out.

        Returns:
            str: The prompt for the player.
        """
        if discussion_history is None:
            discussion_history = ""
        if question is None:
            question = ""

        # Get list of player names (using visible player names)
        player_names = [p.player_name for p in all_players if p.alive]

        # Make sure we're only using player_name (not model_name) for other players
        # This ensures players only know each other by their player names
        player_info = [{"name": p.player_name, "alive": p.alive} for p in all_players]

        # Get the appropriate language, defaulting to English if not supported
        language = self.language if self.language in GAME_RULES else "English"

        # Get game rules for the player's language
        game_rules = GAME_RULES[language]

        if self.role == Role.MAFIA:
            # For Mafia members (using visible player names)
            mafia_names = [
                p.player_name for p in mafia_members if p != self and p.alive
            ]
            mafia_list = f"{', '.join(mafia_names) if mafia_names else 'None (you are the only Mafia left)'}"
            if language == "Spanish":
                mafia_list = f"{', '.join(mafia_names) if mafia_names else 'Ninguno (eres el único miembro de la Mafia que queda)'}"
            elif language == "French":
                mafia_list = f"{', '.join(mafia_names) if mafia_names else 'Aucun (vous êtes le seul membre de la Mafia restant)'}"
            elif language == "Korean":
                mafia_list = f"{', '.join(mafia_names) if mafia_names else '없음 (당신이 유일하게 남은 마피아입니다)'}"

            prompt = PROMPT_TEMPLATES_HIDDEN[language][Role.MAFIA].format(
                model_name=self.player_name,  # Use player_name in prompts
                game_rules=game_rules,
                mafia_members=mafia_list,
                player_names=", ".join(player_names),
                game_state=game_state,
                question=question,
                discussion_history=discussion_history,
            )
        elif self.role == Role.DOCTOR:
            # For Doctor
            prompt = PROMPT_TEMPLATES_HIDDEN[language][Role.DOCTOR].format(
                model_name=self.player_name,  # Use player_name in prompts
                game_rules=game_rules,
                player_names=", ".join(player_names),
                game_state=game_state,
                question=question,
                discussion_history=discussion_history,
            )
        else:  # Role.VILLAGER
            # For Villagers
            prompt = PROMPT_TEMPLATES_HIDDEN[language][Role.VILLAGER].format(
                model_name=self.player_name,  # Use player_name in prompts
                game_rules=game_rules,
                player_names=", ".join(player_names),
                game_state=game_state,
                question=question,
                discussion_history=discussion_history,
            )

        return prompt

    def get_response(self, prompt):
        """
        Get a response from the LLM model using OpenRouter API.

        Args:
            prompt (str): The prompt to send to the model.

        Returns:
            str: The response from the model with private thoughts removed.
        """
        print(f"\n\n{len(prompt)=} {prompt=}\n\n")
        response = get_llm_response(self.model_name, prompt)

        # Remove any <think></think> tags and their contents before sharing with other players
        cleaned_response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL)

        # Clean up any extra whitespace that might have been created
        cleaned_response = re.sub(r"\n\s*\n", "\n\n", cleaned_response)
        cleaned_response = cleaned_response.strip()

        return cleaned_response

    def parse_night_action(self, response, all_players):
        """
        Parse the night action from the player's response.

        Args:
            response (str): The response from the player (already cleaned of thinking tags).
            all_players (list): List of all players in the game.

        Returns:
            tuple: (action_type, target_player) or (None, None) if no valid action.
        """
        if self.role == Role.MAFIA:
            # Look for action pattern based on language
            pattern = ACTION_PATTERNS.get(self.language, ACTION_PATTERNS["English"])[
                Role.MAFIA
            ]
            match = re.search(pattern, response, re.IGNORECASE)

            if match:
                target_name = match.group(1).strip()
                # Find the target player, excluding Mafia members
                target_player = self._find_target_player(
                    target_name, all_players, exclude_mafia=True
                )
                if target_player:
                    return "kill", target_player
            return None, None

        elif self.role == Role.DOCTOR:
            # Look for action pattern based on language
            pattern = ACTION_PATTERNS.get(self.language, ACTION_PATTERNS["English"])[
                Role.DOCTOR
            ]
            match = re.search(pattern, response, re.IGNORECASE)

            if match:
                target_name = match.group(1).strip()
                # Find the target player
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

        Args:
            response (str): The response from the player (already cleaned of thinking tags).
            all_players (list): List of all players in the game.

        Returns:
            Player or None: The player being voted for, or None if no valid vote.
        """
        # Get vote pattern based on language
        pattern = VOTE_PATTERNS.get(self.language, VOTE_PATTERNS["English"])
        match = re.search(pattern, response, re.IGNORECASE)

        if match:
            target_name = match.group(1).strip()
            # Find the target player
            return self._find_target_player(target_name, all_players)
        return None

    def get_confirmation_vote(self, game_state):
        """
        Get a confirmation vote from the player on whether to eliminate another player.

        Args:
            game_state (dict): The current state of the game, including who is up for elimination.

        Returns:
            str: "agree" or "disagree" indicating the player's vote
        """
        player_to_eliminate = game_state["confirmation_vote_for"]
        game_state_str = game_state["game_state"]

        # Get the appropriate language, defaulting to English if not supported
        language = (
            self.language
            if self.language in CONFIRMATION_VOTE_EXPLANATIONS
            else "English"
        )

        # Get confirmation vote explanation for the player's language
        confirmation_explanation = CONFIRMATION_VOTE_EXPLANATIONS[language].format(
            player_to_eliminate=player_to_eliminate
        )

        # Generate prompt based on language
        prompt = CONFIRMATION_VOTE_TEMPLATES[language].format(
            model_name=self.player_name,  # Use player_name in prompts
            player_to_eliminate=player_to_eliminate,
            confirmation_explanation=confirmation_explanation,
            game_state_str=game_state_str,
            thinking_tag=THINKING_TAGS[language],
        )

        response = self.get_response(prompt)

        # Parse the response for agree/disagree based on language
        language = (
            self.language if self.language in CONFIRMATION_VOTE_PATTERNS else "English"
        )
        if re.search(CONFIRMATION_VOTE_PATTERNS[language]["agree"], response.lower()):
            return "agree"
        else:
            return "disagree"
