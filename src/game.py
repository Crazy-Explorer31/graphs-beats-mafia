"""
Game logic for the LLM Mafia Game Competition.
"""

import random
import uuid
from player import Player
from game_templates import Role
import config
from logger import GameLogger, Color
import re
import json
from openrouter import get_llm_response
from game_state import GameStateManager, DiscussionHistory
from game_orchestrator import GameOrchestrator


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
        self.logger.phase_header("Night", self.round_number)

        # Reset protected status
        for player in self.players:
            player.protected = False

        # Get actions from Mafia players
        mafia_targets = []
        for player in self.mafia_players:
            if player.alive:
                # Generate prompt
                game_state = f"{self.get_game_state()} It's night time (Round {self.round_number}). As the Mafia, you MUST choose exactly one player to kill tonight. You cannot skip this action. End your response with ACTION: Kill [player]."
                prompt = player.generate_prompt(
                    game_state,
                    self.get_alive_players(),
                    self.mafia_players,
                    self.discussion_history_without_thinking(),
                )

                # Get response
                response = player.get_response(prompt)
                # Log using player_name as the visible identifier
                self.logger.player_response(
                    player.player_name, "Mafia", response, player.player_name
                )

                # Add to messages using player_name as the speaker identifier
                self.current_round_data["messages"].append(
                    {
                        "speaker": player.player_name,
                        "content": response,
                        "phase": "night",
                        "role": "Mafia",
                        "player_name": player.player_name,
                    }
                )

                # Parse action
                action_type, target = player.parse_night_action(
                    response, self.get_alive_players()
                )

                if action_type == "kill" and target:
                    mafia_targets.append(target)
                    action_text = f"Kill {target.player_name}"
                    # Use player_name as the key for actions
                    self.current_round_data["actions"][player.player_name] = action_text
                    self.logger.player_action(
                        player.player_name, "Mafia", action_text, player.player_name
                    )
                else:
                    self.logger.error(
                        f"Invalid action from {player.player_name} (Mafia)"
                    )
                    self.current_round_data["actions"][
                        player.player_name
                    ] = "Invalid action"

        # Determine Mafia kill target (majority vote)
        kill_target = None
        if mafia_targets:
            # Count votes for each target using player_name
            target_counts = {}
            for target in mafia_targets:
                if target.player_name in target_counts:
                    target_counts[target.player_name] += 1
                else:
                    target_counts[target.player_name] = 1

            # Find target with most votes
            max_votes = 0
            for target_name, votes in target_counts.items():
                if votes > max_votes:
                    max_votes = votes
                    for player in self.get_alive_players():
                        if player.player_name == target_name:
                            kill_target = player
                            break

            # Record the final mafia target using player_name
            if kill_target:
                self.current_round_data["targeted_by_mafia"].append(
                    kill_target.player_name
                )

        # Get action from Doctor
        protected_player = None
        if self.doctor_player and self.doctor_player.alive:
            # Generate prompt with language-specific instructions
            night_instructions = {
                "English": f"It's night time (Round {self.round_number}). As the Doctor, you MUST choose exactly one player to protect from the Mafia tonight. You cannot skip this action. End your response with ACTION: Protect [player].",
                "Spanish": f"Es hora de noche (Ronda {self.round_number}). Como Doctor, DEBES elegir exactamente a un jugador para proteger de la Mafia esta noche. No puedes omitir esta acción. Termina tu respuesta con ACCIÓN: Proteger [jugador].",
                "French": f"C'est la nuit (Tour {self.round_number}). En tant que Docteur, vous DEVEZ choisir exactement un joueur à protéger de la Mafia ce soir. Vous ne pouvez pas ignorer cette action. Terminez votre réponse par ACTION: Protéger [joueur].",
                "Korean": f"밤 시간입니다 (라운드 {self.round_number}). 의사로서, 당신은 오늘 밤 마피아로부터 보호할 플레이어를 정확히 한 명 선택해야 합니다. 이 행동을 건너뛸 수 없습니다. 응답 끝에 행동: 보호하기 [플레이어]를 포함하세요.",
            }

            # Get the appropriate instruction based on the doctor's language
            instruction = night_instructions.get(
                self.doctor_player.language, night_instructions["English"]
            )

            game_state = f"{self.get_game_state()} {instruction}"
            prompt = self.doctor_player.generate_prompt(
                game_state,
                self.get_alive_players(),
                None,
                self.discussion_history_without_thinking(),
            )

            # Get response
            response = self.doctor_player.get_response(prompt)
            # Log using player_name as the visible identifier
            self.logger.player_response(
                self.doctor_player.player_name,
                "Doctor",
                response,
                self.doctor_player.player_name,
            )

            # Add to messages using player_name as the speaker identifier
            self.current_round_data["messages"].append(
                {
                    "speaker": self.doctor_player.player_name,
                    "content": response,
                    "phase": "night",
                    "role": "Doctor",
                    "player_name": self.doctor_player.player_name,
                }
            )

            # Parse action
            action_type, target = self.doctor_player.parse_night_action(
                response, self.get_alive_players()
            )

            if action_type == "protect" and target:
                protected_player = target
                target.protected = True
                action_text = f"Protect {target.player_name}"
                # Use player_name as the key for actions
                self.current_round_data["actions"][
                    self.doctor_player.player_name
                ] = action_text
                # Store protected player by player_name
                self.current_round_data["protected_by_doctor"].append(target.player_name)
                self.logger.player_action(
                    self.doctor_player.player_name,
                    "Doctor",
                    action_text,
                    self.doctor_player.player_name,
                )
            else:
                self.logger.error(
                    f"Invalid action from {self.doctor_player.player_name} (Doctor)"
                )
                self.current_round_data["actions"][
                    self.doctor_player.player_name
                ] = "Invalid action"

        # Process night actions
        eliminated_players = []
        if kill_target and not kill_target.protected:
            kill_target.alive = False
            eliminated_players.append(kill_target)
            # Store eliminated player by player_name
            self.current_round_data["eliminations"].append(kill_target.player_name)
            outcome_text = f"{kill_target.player_name} was killed by the Mafia."
            self.current_round_data["outcome"] = outcome_text
            self.logger.event(outcome_text, Color.RED)
        else:
            if kill_target and kill_target.protected:
                outcome_text = f"The Doctor protected {kill_target.player_name} from the Mafia."
                self.current_round_data["outcome"] = outcome_text
                self.logger.event(outcome_text, Color.BLUE)
            else:
                outcome_text = "No one was killed during the night."
                self.current_round_data["outcome"] = outcome_text
                self.logger.event(outcome_text, Color.YELLOW)

        # Set phase to day
        self.phase = "day"

        return eliminated_players

    def execute_day_phase(self):
        """
        Execute the day phase of the game.

        Returns:
            list: List of eliminated players.
        """
        self.logger.phase_header("Day", self.round_number)

        # Get alive players
        alive_players = self.get_alive_players()

        # Collect messages and votes from all alive players
        messages = []
        votes = {}

        # First round: Discussion without voting
        self.logger.event("Discussion Round - Players share their thoughts", Color.CYAN)
        self._conduct_player_interactions(
            alive_players,
            "day_discussion",
            f"It's day time (Round {self.round_number}). Discuss with other players about who might be Mafia. This is the DISCUSSION PHASE ONLY - DO NOT VOTE YET. You will vote in the next round.",
            messages,
            collect_votes=False,
        )

        # Second round: Discussion with voting
        self.logger.event(
            "Voting Round - Players make their final arguments and vote", Color.CYAN
        )
        self._conduct_player_interactions(
            alive_players,
            "day_voting",
            f"It's now the VOTING PHASE (Round {self.round_number}). Make your final arguments and YOU MUST VOTE to eliminate a suspected Mafia member. End your message with VOTE: [player name].",
            messages,
            collect_votes=True,
            votes=votes,
        )

        # Count votes (votes dict uses player_name -> player_name mapping)
        vote_counts = {}
        vote_details = {}  # Stores who voted for whom (by player_name)
        for voter_name, target_name in votes.items():
            if target_name in vote_counts:
                vote_counts[target_name] += 1
            else:
                vote_counts[target_name] = 1

            # Store voter information for each target (all by player_name)
            if target_name not in vote_details:
                vote_details[target_name] = []
            vote_details[target_name].append(voter_name)

        # Find player with most votes (using player_name throughout)
        max_votes = 0
        eliminated_player = None

        for target_name, vote_count in vote_counts.items():
            if vote_count > max_votes:
                max_votes = vote_count
                for player in alive_players:
                    if player.player_name == target_name:
                        eliminated_player = player
                        break

        # Eliminate player with most votes
        eliminated_players = []
        if eliminated_player:
            # Get confirmation vote before elimination
            is_confirmed, confirmation_votes = self.get_confirmation_vote(
                eliminated_player
            )

            # Store confirmation vote details in the round data
            self.current_round_data["confirmation_votes"] = confirmation_votes

            if not is_confirmed:
                confirmation_text = f"The elimination of {eliminated_player.player_name} was rejected by the town."
                self.current_round_data["outcome"] += f" {confirmation_text}"
                self.logger.event(confirmation_text, Color.YELLOW)

                # No elimination if confirmation vote fails
                eliminated_player = None
                eliminated_players = []

                # Store vote information even if no one was eliminated
                self.current_round_data["vote_counts"] = vote_counts
                self.current_round_data["vote_details"] = vote_details
            else:
                # Get last words from the player before elimination
                last_words = self.get_last_words(
                    eliminated_player, vote_counts[eliminated_player.player_name]
                )

                eliminated_player.alive = False
                eliminated_players.append(eliminated_player)
                # Store eliminated player by player_name
                self.current_round_data["eliminations"].append(
                    eliminated_player.player_name
                )
                # Track players eliminated by voting using player_name
                self.current_round_data["eliminated_by_vote"] = [
                    eliminated_player.player_name
                ]

                # Store vote details in the round data
                self.current_round_data["vote_counts"] = vote_counts
                self.current_round_data["vote_details"] = vote_details

                # Include vote count in the outcome text (use player_name only)
                outcome_text = f"{eliminated_player.player_name} was eliminated by vote with {vote_counts[eliminated_player.player_name]} votes."
                self.current_round_data["outcome"] += f" {outcome_text}"
                self.logger.event(outcome_text, Color.YELLOW)

                # Add last words to the outcome and discussion history
                if last_words:
                    last_words_text = f"{eliminated_player.player_name}'s last words: \"{last_words}\""
                    self.current_round_data["last_words"] = last_words
                    self.logger.event(last_words_text, Color.CYAN)
                    # Add last words to discussion history
                    self.discussion_history += (
                        f"{eliminated_player.player_name}: {last_words}\n\n"
                    )
                    self.discussion_history_last_round += (
                        f"{eliminated_player.player_name}: {last_words}\n\n"
                    )
                    # Add to messages using player_name as speaker
                    self.current_round_data["messages"].append(
                        {
                            "speaker": eliminated_player.player_name,
                            "content": last_words,
                            "phase": "day",
                            "role": eliminated_player.role.value,
                            "type": "last_words",
                            "player_name": eliminated_player.player_name,
                        }
                    )

                # Log who voted for the eliminated player (all by player_name)
                voters = vote_details.get(eliminated_player.player_name, [])
                if voters:
                    voter_text = f"Voted by: {', '.join(voters)}"
                    self.current_round_data["voters"] = voters
                    self.logger.event(voter_text, Color.YELLOW)
        else:
            outcome_text = "No one was eliminated by vote."
            self.current_round_data["outcome"] += f" {outcome_text}"
            self.logger.event(outcome_text, Color.YELLOW)

            # Still store vote information even if no one was eliminated
            self.current_round_data["vote_counts"] = vote_counts
            self.current_round_data["vote_details"] = vote_details

        # Set phase to night and increment round
        self.phase = "night"
        self.state.add_round_data(self.current_round_data)
        self.round_number += 1
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

        return eliminated_players

    def _conduct_player_interactions(
        self,
        alive_players,
        phase_type,
        instruction,
        messages,
        collect_votes=False,
        votes=None,
    ):
        """
        Conduct interactions with all alive players during the day phase.

        Args:
            alive_players (list): List of alive players
            phase_type (str): Type of phase (day_discussion or day_voting)
            instruction (str): Specific instruction for this interaction round
            messages (list): List to collect all messages
            collect_votes (bool): Whether to collect votes in this round
            votes (dict): Dictionary to store votes (player_name -> player_name) if collect_votes is True
        """
        self.discussion_history_last_round = ""
        for player in alive_players:
            # Generate prompt
            game_state = f"{self.get_game_state()} {instruction}"

            # Add special instruction for doctor during day phase
            if player.role == Role.DOCTOR:
                day_warnings = {
                    "English": " IMPORTANT: This is the DAY phase. Do NOT use your protection ability now. Only use ACTION: Protect during night phase.",
                    "Spanish": " IMPORTANTE: Esta es la fase DIURNA. NO uses tu habilidad de protección ahora. Solo usa ACCIÓN: Proteger durante la fase nocturna.",
                    "French": " IMPORTANT: C'est la phase de JOUR. N'utilisez PAS votre capacité de protection maintenant. Utilisez ACTION: Protéger uniquement pendant la phase de nuit.",
                    "Korean": " 중요: 지금은 낮 단계입니다. 지금은 보호 능력을 사용하지 마세요. 행동: 보호하기는 밤 단계에서만 사용하세요.",
                }

                # Get the appropriate warning based on the doctor's language
                warning = day_warnings.get(player.language, day_warnings["English"])
                game_state += warning

            # Add special instruction for mafia players during day phase
            elif player.role == Role.MAFIA:
                day_warnings = {
                    "English": " IMPORTANT: This is the DAY phase. Do NOT use 'ACTION: Kill' now. Instead, use 'VOTE: [player]' to vote like other villagers.",
                    "Spanish": " IMPORTANTE: Esta es la fase DIURNA. NO uses 'ACCIÓN: Matar' ahora. En su lugar, usa 'VOTO: [jugador]' para votar como los demás aldeanos.",
                    "French": " IMPORTANT: C'est la phase de JOUR. N'utilisez PAS 'ACTION: Tuer' maintenant. À la place, utilisez 'VOTE: [joueur]' pour voter comme les autres villageois.",
                    "Korean": " 중요: 지금은 낮 단계입니다. '행동: 죽이기'를 사용하지 마세요. 대신 다른 마을 사람들처럼 '투표: [플레이어]'를 사용하여 투표하세요.",
                }

                # Get the appropriate warning based on the mafia player's language
                warning = day_warnings.get(player.language, day_warnings["English"])
                game_state += warning

            # Add voting reminder for all players during voting phase
            if phase_type == "day_voting":
                voting_reminders = {
                    "English": " REMINDER: This is the VOTING PHASE. You MUST end your message with 'VOTE: [player]' to cast your vote.",
                    "Spanish": " RECORDATORIO: Esta es la fase de VOTACIÓN. DEBES terminar tu mensaje con 'VOTO: [jugador]' para emitir tu voto.",
                    "French": " RAPPEL: C'est la phase de VOTE. Vous DEVEZ terminer votre message par 'VOTE: [joueur]' pour exprimer votre vote.",
                    "Korean": " 알림: 지금은 투표 단계입니다. 반드시 메시지 끝에 '투표: [플레이어]'를 포함하여 투표해야 합니다.",
                }

                # Get the appropriate reminder based on the player's language
                reminder = voting_reminders.get(
                    player.language, voting_reminders["English"]
                )
                game_state += reminder

            prompt = player.generate_prompt(
                game_state,
                alive_players,
                self.mafia_players if player.role == Role.MAFIA else None,
                self.discussion_history_without_thinking(),
            )

            # Get response
            response = player.get_response(prompt)
            # Log using player_name as the visible identifier
            self.logger.player_response(
                player.player_name, player.role.value, response, player.player_name
            )

            # Add to messages using player_name as the speaker identifier
            messages.append(
                {
                    "speaker": player.player_name,
                    "content": response,
                    "player_name": player.player_name,
                }
            )
            self.current_round_data["messages"].append(
                {
                    "speaker": player.player_name,
                    "content": response,
                    "phase": phase_type,
                    "role": player.role.value,
                    "player_name": player.player_name,
                }
            )

            # Parse vote if in voting round; store as player_name -> player_name
            if collect_votes and votes is not None:
                vote_target = player.parse_day_vote(response, alive_players)
                if vote_target:
                    votes[player.player_name] = vote_target.player_name
                    action_text = f"Vote {vote_target.player_name}"
                    # Use player_name as the key for actions
                    self.current_round_data["actions"][player.player_name] = action_text
                    self.logger.player_action(
                        player.player_name,
                        player.role.value,
                        action_text,
                        player.player_name,
                    )
                else:
                    self.logger.warning(
                        f"{player.player_name} failed to cast a valid vote during voting phase"
                    )
                    self.current_round_data["actions"][
                        player.player_name
                    ] = "Invalid vote"

            # Generate a per-phrase relationship graph for this player's message
            # and append it to their graph_sequence. This is done after response
            # parsing but before appending to the discussion history.
            try:
                player.generate_per_phrase_graph(
                    message=response,
                    alive_players=alive_players,
                    current_round=self.round_number,
                    phase=phase_type,
                )
            except Exception as e:
                self.logger.warning(
                    f"[PhraseGraph] Failed to generate per-phrase graph for "
                    f"{player.player_name}: {e}"
                )

            # Update discussion history using player_name as the speaker
            self.discussion_history += f"{player.player_name}: {response}\n\n"
            self.discussion_history_last_round += f"{player.player_name}: {response}\n\n"

    def get_last_words(self, player, vote_count):
        """
        Get the last words from a player who is about to be eliminated.

        Args:
            player (Player): The player who is about to be eliminated.
            vote_count (int): The number of votes against the player.

        Returns:
            str: The player's last words.
        """
        self.logger.event(
            f"Getting last words from {player.player_name}...",
            Color.CYAN,
        )

        # Generate prompt for last words
        game_state = f"{self.get_game_state()} You have been voted out with {vote_count} votes and will be eliminated. Share your final thoughts before leaving the game."
        prompt = player.generate_prompt(
            game_state,
            self.get_alive_players(),
            self.mafia_players if player.role == Role.MAFIA else None,
            self.discussion_history_without_thinking(),
        )

        # Get response
        response = player.get_response(prompt)
        self.logger.player_response(
            player.player_name,
            f"{player.role.value} (Last Words)",
            response,
            player.player_name,
        )

        return response

    def get_confirmation_vote(self, player_to_eliminate):
        """
        Get confirmation votes from all alive players on whether to eliminate a player.

        Args:
            player_to_eliminate: The player who is proposed for elimination

        Returns:
            tuple: (bool, dict) - Whether the elimination is confirmed and the vote details
        """
        alive_players = self.get_alive_players()

        # Don't include the player to be eliminated in the voting
        voting_players = [p for p in alive_players if p != player_to_eliminate]

        self.logger.event(
            f"Confirmation vote for eliminating {player_to_eliminate.player_name}",
            Color.YELLOW,
        )

        # Collect votes (identified by player_name)
        confirmation_votes = {"agree": [], "disagree": []}

        for player in voting_players:
            # Prepare game state for the player
            game_state_str = self.get_game_state()
            # Create a dictionary with the game state and the player to eliminate (by player_name)
            player_state = {
                "game_state": game_state_str,
                "confirmation_vote_for": player_to_eliminate.player_name,
                "confirmation_vote_for_model": player_to_eliminate.model_name,
            }

            # Get player's vote
            vote = player.get_confirmation_vote(player_state, self.players, self.discussion_history_without_thinking())

            # Validate and record vote using player_name
            if vote.lower() in ["agree", "yes", "confirm", "true"]:
                confirmation_votes["agree"].append(player.player_name)
                self.logger.event(
                    f"{player.player_name} voted to CONFIRM elimination",
                    Color.GREEN,
                )
            else:
                confirmation_votes["disagree"].append(player.player_name)
                self.logger.event(
                    f"{player.player_name} voted to REJECT elimination",
                    Color.RED,
                )

        # Check if more than half of the voting players agreed
        is_confirmed = len(confirmation_votes["agree"]) > len(voting_players) / 2

        return is_confirmed, confirmation_votes

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
