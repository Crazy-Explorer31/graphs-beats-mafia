import config
from logger import Color
from game_templates import Role

class NightExecutor:
    def __init__(self, game):
        self.game = game

    def execute(self):
        self.game.logger.phase_header("Night", self.game.round_number)
        for player in self.game.players:
            player.protected = False

        mafia_targets = []
        for player in self.game.mafia_players:
            if not player.alive:
                continue
            game_state = (
                f"{self.game.get_game_state()} It's night time (Round {self.game.round_number}). "
                "As the Mafia, you MUST choose exactly one player to kill tonight. "
                "You cannot skip this action. End your response with ACTION: Kill [player]."
            )
            prompt = player.generate_prompt(
                game_state,
                self.game.get_alive_players(),
                self.game.mafia_players,
                self.game.discussion_history_without_thinking(),
            )
            response = player.get_response(prompt)
            self.game.logger.player_response(
                player.model_name, "Mafia", response, player_name=player.player_name
            )
            self.game.current_round_data["messages"].append({
                "speaker": player.player_name,
                "content": response,
                "phase": "night",
                "role": "Mafia",
                "player_name": player.player_name,
            })
            action_type, target = player.parse_night_action(response, self.game.get_alive_players())
            if action_type == "kill" and target:
                mafia_targets.append(target)
                action_text = f"Kill {target.player_name}"
                self.game.current_round_data["actions"][player.player_name] = action_text
                self.game.logger.player_action(
                    player.model_name, "Mafia", action_text, player_name=player.player_name
                )
            else:
                self.game.logger.error(f"Invalid action from {player.player_name} (Mafia)")
                self.game.current_round_data["actions"][player.player_name] = "Invalid action"

        kill_target = None
        if mafia_targets:
            target_counts = {}
            for target in mafia_targets:
                target_counts[target.player_name] = target_counts.get(target.player_name, 0) + 1
            max_votes = 0
            for target_name, votes in target_counts.items():
                if votes > max_votes:
                    max_votes = votes
                    for p in self.game.get_alive_players():
                        if p.player_name == target_name:
                            kill_target = p
                            break
            if kill_target:
                self.game.current_round_data["targeted_by_mafia"].append(kill_target.player_name)

        protected_player = None
        if self.game.doctor_player and self.game.doctor_player.alive:
            night_instructions = {
                "English": (
                    f"It's night time (Round {self.game.round_number}). "
                    "As the Doctor, you MUST choose exactly one player to protect from the Mafia tonight. "
                    "You cannot skip this action. End your response with ACTION: Protect [player]."
                ),
                "Spanish": (
                    f"Es hora de noche (Ronda {self.game.round_number}). "
                    "Como Doctor, DEBES elegir exactamente a un jugador para proteger de la Mafia esta noche. "
                    "No puedes omitir esta acción. Termina tu respuesta con ACCIÓN: Proteger [jugador]."
                ),
                "French": (
                    f"C'est la nuit (Tour {self.game.round_number}). "
                    "En tant que Docteur, vous DEVEZ choisir exactement un joueur à protéger de la Mafia ce soir. "
                    "Vous ne pouvez pas ignorer cette action. Terminez votre réponse par ACTION: Protéger [joueur]."
                ),
                "Korean": (
                    f"밤 시간입니다 (라운드 {self.game.round_number}). "
                    "의사로서, 당신은 오늘 밤 마피아로부터 보호할 플레이어를 정확히 한 명 선택해야 합니다. "
                    "이 행동을 건너뛸 수 없습니다. 응답 끝에 행동: 보호하기 [플레이어]를 포함하세요."
                ),
            }
            instruction = night_instructions.get(
                self.game.doctor_player.language, night_instructions["English"]
            )
            game_state = f"{self.game.get_game_state()} {instruction}"
            prompt = self.game.doctor_player.generate_prompt(
                game_state,
                self.game.get_alive_players(),
                None,
                self.game.discussion_history_without_thinking(),
            )
            response = self.game.doctor_player.get_response(prompt)
            self.game.logger.player_response(
                self.game.doctor_player.model_name,
                "Doctor",
                response,
                player_name=self.game.doctor_player.player_name,
            )
            self.game.current_round_data["messages"].append({
                "speaker": self.game.doctor_player.player_name,
                "content": response,
                "phase": "night",
                "role": "Doctor",
                "player_name": self.game.doctor_player.player_name,
            })
            action_type, target = self.game.doctor_player.parse_night_action(
                response, self.game.get_alive_players()
            )
            if action_type == "protect" and target:
                protected_player = target
                target.protected = True
                action_text = f"Protect {target.player_name}"
                self.game.current_round_data["actions"][self.game.doctor_player.player_name] = action_text
                self.game.current_round_data["protected_by_doctor"].append(target.player_name)
                self.game.logger.player_action(
                    self.game.doctor_player.model_name,
                    "Doctor",
                    action_text,
                    player_name=self.game.doctor_player.player_name,
                )
            else:
                self.game.logger.error(
                    f"Invalid action from {self.game.doctor_player.player_name} (Doctor)"
                )
                self.game.current_round_data["actions"][self.game.doctor_player.player_name] = "Invalid action"

        eliminated_players = []
        if kill_target and not kill_target.protected:
            kill_target.alive = False
            eliminated_players.append(kill_target)
            self.game.current_round_data["eliminations"].append(kill_target.player_name)
            outcome = f"{kill_target.player_name} was killed by the Mafia."
            self.game.current_round_data["outcome"] = outcome
            self.game.logger.event(outcome, Color.RED)
        else:
            if kill_target and kill_target.protected:
                outcome = f"The Doctor protected {kill_target.player_name} from the Mafia."
                self.game.current_round_data["outcome"] = outcome
                self.game.logger.event(outcome, Color.BLUE)
            else:
                outcome = "No one was killed during the night."
                self.game.current_round_data["outcome"] = outcome
                self.game.logger.event(outcome, Color.YELLOW)

        self.game.phase = "day"
        return eliminated_players


class DayExecutor:
    def __init__(self, game):
        self.game = game

    def execute(self):
        self.game.logger.phase_header("Day", self.game.round_number)
        alive_players = self.game.get_alive_players()
        messages = []
        votes = {}

        self.game.logger.event("Discussion Round - Players share their thoughts", Color.CYAN)
        self._conduct_player_interactions(
            alive_players,
            "day_discussion",
            f"It's day time (Round {self.game.round_number}). Discuss with other players about who might be Mafia. "
            "This is the DISCUSSION PHASE ONLY - DO NOT VOTE YET. You will vote in the next round.",
            messages,
            collect_votes=False,
        )

        self.game.logger.event("Voting Round - Players make their final arguments and vote", Color.CYAN)
        self._conduct_player_interactions(
            alive_players,
            "day_voting",
            f"It's now the VOTING PHASE (Round {self.game.round_number}). Make your final arguments and "
            "YOU MUST VOTE to eliminate a suspected Mafia member. End your message with VOTE: [player name].",
            messages,
            collect_votes=True,
            votes=votes,
        )

        vote_counts = {}
        vote_details = {}
        for voter_name, target_name in votes.items():
            vote_counts[target_name] = vote_counts.get(target_name, 0) + 1
            if target_name not in vote_details:
                vote_details[target_name] = []
            vote_details[target_name].append(voter_name)

        max_votes = 0
        eliminated_player = None
        for target_name, cnt in vote_counts.items():
            if cnt > max_votes:
                max_votes = cnt
                for player in alive_players:
                    if player.player_name == target_name:
                        eliminated_player = player
                        break

        eliminated_players = []
        if eliminated_player:
            is_confirmed, confirmation_votes = self.get_confirmation_vote(eliminated_player)
            self.game.current_round_data["confirmation_votes"] = confirmation_votes
            if not is_confirmed:
                confirm_text = f"The elimination of {eliminated_player.player_name} was rejected by the town."
                self.game.current_round_data["outcome"] += f" {confirm_text}"
                self.game.logger.event(confirm_text, Color.YELLOW)
                eliminated_player = None
                eliminated_players = []
                self.game.current_round_data["vote_counts"] = vote_counts
                self.game.current_round_data["vote_details"] = vote_details
            else:
                last_words = self.get_last_words(
                    eliminated_player, vote_counts[eliminated_player.player_name]
                )
                eliminated_player.alive = False
                eliminated_players.append(eliminated_player)
                self.game.current_round_data["eliminations"].append(eliminated_player.player_name)
                self.game.current_round_data["eliminated_by_vote"] = [eliminated_player.player_name]
                self.game.current_round_data["vote_counts"] = vote_counts
                self.game.current_round_data["vote_details"] = vote_details

                outcome_text = (
                    f"{eliminated_player.player_name} was eliminated by vote with "
                    f"{vote_counts[eliminated_player.player_name]} votes."
                )
                self.game.current_round_data["outcome"] += f" {outcome_text}"
                self.game.logger.event(outcome_text, Color.YELLOW)

                if last_words:
                    lw_text = (
                        f"{eliminated_player.player_name}'s last words: \"{last_words}\""
                    )
                    self.game.current_round_data["last_words"] = last_words
                    self.game.logger.event(lw_text, Color.CYAN)
                    self.game.discussion_history += f"{eliminated_player.player_name}: {last_words}\n\n"
                    self.game.discussion_history_last_round += f"{eliminated_player.player_name}: {last_words}\n\n"
                    self.game.current_round_data["messages"].append({
                        "speaker": eliminated_player.player_name,
                        "content": last_words,
                        "phase": "day",
                        "role": eliminated_player.role.value,
                        "type": "last_words",
                        "player_name": eliminated_player.player_name,
                    })

                voters_list = vote_details.get(eliminated_player.player_name, [])
                if voters_list:
                    voter_text = f"Voted by: {', '.join(voters_list)}"
                    self.game.current_round_data["voters"] = voters_list
                    self.game.logger.event(voter_text, Color.YELLOW)
        else:
            outcome_text = "No one was eliminated by vote."
            self.game.current_round_data["outcome"] += f" {outcome_text}"
            self.game.logger.event(outcome_text, Color.YELLOW)
            self.game.current_round_data["vote_counts"] = vote_counts
            self.game.current_round_data["vote_details"] = vote_details

        self.game.phase = "night"
        self.game.state.add_round_data(self.game.current_round_data)
        self.game.round_number += 1
        self.game.current_round_data = {
            "round_number": self.game.round_number,
            "messages": [],
            "actions": {},
            "eliminations": [],
            "eliminated_by_vote": [],
            "targeted_by_mafia": [],
            "protected_by_doctor": [],
            "outcome": "",
        }
        return eliminated_players

    def _conduct_player_interactions(self, alive_players, phase_type, instruction, messages, collect_votes=False, votes=None):
        self.game.discussion_history_last_round = ""
        for player in alive_players:
            game_state = f"{self.game.get_game_state()} {instruction}"

            if player.role == Role.DOCTOR:
                day_warnings = {
                    "English": " IMPORTANT: This is the DAY phase. Do NOT use your protection ability now. Only use ACTION: Protect during night phase.",
                    "Spanish": " IMPORTANTE: Esta es la fase DIURNA. NO uses tu habilidad de protección ahora. Solo usa ACCIÓN: Proteger durante la fase nocturna.",
                    "French": " IMPORTANT: C'est la phase de JOUR. N'utilisez PAS votre capacité de protection maintenant. Utilisez ACTION: Protéger uniquement pendant la phase de nuit.",
                    "Korean": " 중요: 지금은 낮 단계입니다. 지금은 보호 능력을 사용하지 마세요. 행동: 보호하기는 밤 단계에서만 사용하세요.",
                }
                warning = day_warnings.get(player.language, day_warnings["English"])
                game_state += warning
            elif player.role == Role.MAFIA:
                day_warnings = {
                    "English": " IMPORTANT: This is the DAY phase. Do NOT use 'ACTION: Kill' now. Instead, use 'VOTE: [player]' to vote like other villagers.",
                    "Spanish": " IMPORTANTE: Esta es la fase DIURNA. NO uses 'ACCIÓN: Matar' ahora. En su lugar, usa 'VOTO: [jugador]' para votar como los demás aldeanos.",
                    "French": " IMPORTANT: C'est la phase de JOUR. N'utilisez PAS 'ACTION: Tuer' maintenant. À la place, utilisez 'VOTE: [joueur]' pour voter comme les autres villageois.",
                    "Korean": " 중요: 지금은 낮 단계입니다. '행동: 죽이기'를 사용하지 마세요. 대신 다른 마을 사람들처럼 '투표: [플레이어]'를 사용하여 투표하세요.",
                }
                warning = day_warnings.get(player.language, day_warnings["English"])
                game_state += warning

            if phase_type == "day_voting":
                voting_reminders = {
                    "English": " REMINDER: This is the VOTING PHASE. You MUST end your message with 'VOTE: [player]' to cast your vote.",
                    "Spanish": " RECORDATORIO: Esta es la fase de VOTACIÓN. DEBES terminar tu mensaje con 'VOTO: [jugador]' para emitir tu voto.",
                    "French": " RAPPEL: C'est la phase de VOTE. Vous DEVEZ terminer votre message par 'VOTE: [joueur]' pour exprimer votre vote.",
                    "Korean": " 알림: 지금은 투표 단계입니다. 반드시 메시지 끝에 '투표: [플레이어]'를 포함하여 투표해야 합니다.",
                }
                reminder = voting_reminders.get(player.language, voting_reminders["English"])
                game_state += reminder

            prompt = player.generate_prompt(
                game_state,
                alive_players,
                self.game.mafia_players if player.role == Role.MAFIA else None,
                self.game.discussion_history_without_thinking(),
            )
            response = player.get_response(prompt)
            self.game.logger.player_response(
                player.model_name, player.role.value, response, player_name=player.player_name
            )
            messages.append({
                "speaker": player.player_name,
                "content": response,
                "player_name": player.player_name,
            })
            self.game.current_round_data["messages"].append({
                "speaker": player.player_name,
                "content": response,
                "phase": phase_type,
                "role": player.role.value,
                "player_name": player.player_name,
            })

            if collect_votes and votes is not None:
                vote_target = player.parse_day_vote(response, alive_players)
                if vote_target:
                    votes[player.player_name] = vote_target.player_name
                    action_text = f"Vote {vote_target.player_name}"
                    self.game.current_round_data["actions"][player.player_name] = action_text
                    self.game.logger.player_action(
                        player.model_name,
                        player.role.value,
                        action_text,
                        player_name=player.player_name,
                    )
                else:
                    self.game.logger.warning(
                        f"{player.player_name} failed to cast a valid vote during voting phase"
                    )
                    self.game.current_round_data["actions"][player.player_name] = "Invalid vote"

            # Update per-phrase graphs for each alive player that has phrase graph enabled
            for listener in alive_players:
                if listener.use_phrase_graph:
                    try:
                        listener.generate_per_phrase_graph(
                            message=response,
                            speaker_name=player.player_name,
                            alive_players=alive_players,
                            current_round=self.game.round_number,
                            phase=phase_type,
                        )
                        print(f'{listener.player_name}: add phrase_graph')
                        print(listener.graph_sequence)
                    except Exception as e:
                        self.game.logger.warning(
                            f"[PhraseGraph] Failed to generate per-phrase graph for "
                            f"{listener.player_name} on message from {player.player_name}: {e}"
                        )

            self.game.discussion_history += f"{player.player_name}: {response}\n\n"
            self.game.discussion_history_last_round += f"{player.player_name}: {response}\n\n"

    def get_last_words(self, player, vote_count):
        self.game.logger.event(
            f"Getting last words from {player.player_name}...",
            Color.CYAN,
        )
        game_state = (
            f"{self.game.get_game_state()} You are a {player.role.value}. "
            f"You have been voted out with {vote_count} votes "
            "and will be eliminated. Share your final thoughts before leaving the game."
        )
        prompt = player.generate_prompt(
            game_state,
            self.game.get_alive_players(),
            self.game.mafia_players if player.role == Role.MAFIA else None,
            self.game.discussion_history_without_thinking(),
        )
        response = player.get_response(prompt)
        self.game.logger.player_response(
            player.model_name,
            f"{player.role.value} (Last Words)",
            response,
            player_name=player.player_name,
        )
        return response

    def get_confirmation_vote(self, player_to_eliminate):
        alive_players = self.game.get_alive_players()
        voting_players = [p for p in alive_players if p != player_to_eliminate]
        self.game.logger.event(
            f"Confirmation vote for eliminating {player_to_eliminate.player_name}",
            Color.YELLOW,
        )
        confirmation_votes = {"agree": [], "disagree": []}
        for player in voting_players:
            game_state_str = self.game.get_game_state()
            player_state = {
                "game_state": game_state_str,
                "confirmation_vote_for": player_to_eliminate.player_name,
                "confirmation_vote_for_model": player_to_eliminate.model_name,
            }
            vote = player.get_confirmation_vote(
                player_state, self.game.players, self.game.discussion_history_without_thinking()
            )
            if vote.lower() in ["agree", "yes", "confirm", "true"]:
                confirmation_votes["agree"].append(player.player_name)
                self.game.logger.event(
                    f"{player.player_name} voted to CONFIRM elimination",
                    Color.GREEN,
                )
            else:
                confirmation_votes["disagree"].append(player.player_name)
                self.game.logger.event(
                    f"{player.player_name} voted to REJECT elimination",
                    Color.RED,
                )
        is_confirmed = len(confirmation_votes["agree"]) > len(voting_players) / 2
        return is_confirmed, confirmation_votes
