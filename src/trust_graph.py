"""
TrustGraph class for the LLM Mafia Game Competition.
Encapsulates all logic for the subjective directed trust graph.
"""
import json
import re
import networkx as nx
import config
from game_templates import Role
from openrouter import get_llm_response


class TrustGraph:
    """
    Encapsulates the subjective directed trust graph for a single player.
    All nodes are keyed by player_name.
    """

    def __init__(self, player_name, role, all_players):
        """
        Initialize the trust graph for the given player.

        Args:
            player_name (str): The visible name of the owning player.
            role (Role): The role of the owning player.
            all_players (list): List of all Player objects in the game.
        """
        self.player_name = player_name
        self.role = role
        self.graph = None
        self.init_graph(all_players)

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------

    def init_graph(self, all_players) -> nx.DiGraph:
        """
        Build the initial subjective directed trust graph.

        Vertices: all players identified by player_name with their role probabilities.
        Edges: trust values between players (-1 to 1).
        The owning player knows their own role.
        Mafia players know each other at game start.
        """
        G = nx.DiGraph()

        # Get total counts from config
        total_players = config.PLAYERS_PER_GAME
        mafia_count = config.MAFIA_COUNT
        doctor_count = config.DOCTOR_COUNT
        villager_count = total_players - mafia_count - doctor_count

        # Identify all mafia players by player_name
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

        # Add all players as nodes, identified by player_name
        for player in all_players:
            if player.player_name == self.player_name:
                # Self: know exact role
                role_probs = {
                    "Mafia": 1.0 if self.role.value == "Mafia" else 0.0,
                    "Villager": 1.0 if self.role.value == "Villager" else 0.0,
                    "Doctor": 1.0 if self.role.value == "Doctor" else 0.0,
                }
                self_trust = 1.0
            else:
                # Mafia players know each other by player_name
                if self.role.value == "Mafia" and player.player_name in all_mafia_players:
                    # This is another mafia member — we know their exact role
                    role_probs = {"Mafia": 1.0, "Villager": 0.0, "Doctor": 0.0}
                else:
                    # Others: use calculated probabilities
                    role_probs = {
                        "Mafia": other_mafia_prob,
                        "Villager": other_villager_prob,
                        "Doctor": other_doctor_prob,
                    }
                self_trust = 0.0

            G.add_node(
                player.player_name,
                role_probabilities=role_probs,
                alive=True,
                is_self=(player.player_name == self.player_name),
                actual_role=player.role.value if player.player_name == self.player_name else None,
            )

            # Self-loop for self-trust
            G.add_edge(
                player.player_name,
                player.player_name,
                trust=self_trust,
                evidence=[],
                last_updated=0,
            )

        # Initialize trust edges between all players (identified by player_name)
        for player1 in all_players:
            for player2 in all_players:
                if player1.player_name != player2.player_name:
                    # Mafia trust each other more at start
                    initial_trust = 0.0
                    if (
                        self.role.value == "Mafia"
                        and player1.player_name in all_mafia_players
                        and player2.player_name in all_mafia_players
                    ):
                        initial_trust = 0.5  # Moderate trust between mafia members

                    G.add_edge(
                        player1.player_name,
                        player2.player_name,
                        trust=initial_trust,
                        evidence=[],
                        last_updated=0,
                    )

        self.graph = G
        return G

    # ------------------------------------------------------------------
    # Graph read helpers
    # ------------------------------------------------------------------

    def get_trust(self, from_player: str, to_player: str) -> float:
        """
        Return the trust value on the edge from_player → to_player.

        Args:
            from_player (str): player_name of the source node.
            to_player (str): player_name of the target node.

        Returns:
            float: Trust value, or 0.0 if edge does not exist.
        """
        if self.graph is None:
            return 0.0
        try:
            return self.graph[from_player][to_player]["trust"]
        except KeyError:
            return 0.0

    def get_role_probabilities(self, player_name: str) -> dict:
        """
        Return the role probability dict for the given player.

        Args:
            player_name (str): The player whose probabilities to retrieve.

        Returns:
            dict: e.g. {"Mafia": 0.2, "Villager": 0.6, "Doctor": 0.2}
        """
        if self.graph is None or player_name not in self.graph.nodes:
            return {}
        return self.graph.nodes[player_name].get("role_probabilities", {})

    # ------------------------------------------------------------------
    # Graph mutation helpers
    # ------------------------------------------------------------------

    def mark_dead(self, player_name: str) -> None:
        """
        Mark a player as dead/eliminated in the graph.

        Args:
            player_name (str): The player_name of the eliminated player.
        """
        if self.graph is not None and player_name in self.graph.nodes:
            self.graph.nodes[player_name]["alive"] = False

    # ------------------------------------------------------------------
    # Prompt conversion
    # ------------------------------------------------------------------

    def to_prompt(self, all_players) -> str:
        """
        Convert the subjective graph to a compact text description for an LLM prompt.
        All players are referenced by player_name.

        Args:
            all_players (list): List of all Player objects (used to filter alive players).

        Returns:
            str: Human-readable description of the trust graph.
        """
        if self.graph is None:
            return ""

        alive_players = [p for p in all_players if p.alive]

        lines = ["YOUR TRUST GRAPH:"]

        # My trust in others
        my_trust = []
        for player in alive_players:
            if player.player_name == self.player_name:
                continue

            trust = self.graph[self.player_name][player.player_name]["trust"]
            probs = self.graph.nodes[player.player_name]["role_probabilities"]

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

            if self.role == Role.MAFIA:
                is_mafia = probs.get("Mafia", 0.0) == 1.0
                role_info = f"Role: {'Mafia' if is_mafia else 'Non-mafia (Villager/Doctor)'}"
                my_trust.append(f"  - {player.player_name}: {trust_level}, {role_info}")
            else:
                mafia_prob = probs.get("Mafia", 0.0)
                my_trust.append(
                    f"  - {player.player_name}: {trust_level}, Mafia prob: {mafia_prob:.0%}"
                )

        if my_trust:
            lines.append("My trust in others:")
            lines.extend(my_trust)

        # Summary: most trusted and suspicious
        trusted = []
        suspicious = []

        for player in alive_players:
            if player.player_name == self.player_name:
                continue

            trust = self.graph[self.player_name][player.player_name]["trust"]
            probs = self.graph.nodes[player.player_name]["role_probabilities"]

            if self.role == Role.MAFIA:
                is_mafia = probs.get("Mafia", 0.0) == 1.0
                if not is_mafia:
                    suspicion_value = -trust
                    if suspicion_value > 0.1:
                        suspicious.append((player.player_name, suspicion_value))
            else:
                suspicion_value = probs.get("Mafia", 0.0)
                if suspicion_value > 0.4:
                    suspicious.append((player.player_name, suspicion_value))

            if trust > 0.3:
                trusted.append((player.player_name, trust))

        trusted.sort(key=lambda x: x[1], reverse=True)
        suspicious.sort(key=lambda x: x[1], reverse=True)

        if trusted:
            top_trusted = ", ".join([name for name, _ in trusted[:2]])
            lines.append(f"\nMost trusted: {top_trusted}")

        if suspicious:
            if self.role == Role.MAFIA:
                top_suspicious = ", ".join([name for name, _ in suspicious[:2]])
                lines.append(
                    f"Most distrustful (suspicious non-mafia): {top_suspicious}"
                )
            else:
                top_suspicious = ", ".join([name for name, _ in suspicious[:2]])
                lines.append(f"Most suspicious (likely Mafia): {top_suspicious}")

        # Key mutual relationships (referenced by player_name)
        mutual_relations = []
        for player1 in alive_players:
            for player2 in alive_players:
                if (
                    player1.player_name == self.player_name
                    or player2.player_name == self.player_name
                    or player1.player_name == player2.player_name
                ):
                    continue

                trust1 = self.graph[player1.player_name][player2.player_name]["trust"]
                trust2 = self.graph[player2.player_name][player1.player_name]["trust"]

                if trust1 > 0.4 and trust2 > 0.4:
                    mutual_relations.append(
                        f"  - {player1.player_name} ↔ {player2.player_name} trust each other"
                    )
                elif trust1 < -0.3 and trust2 < -0.3:
                    mutual_relations.append(
                        f"  - {player1.player_name} ↔ {player2.player_name} distrust each other"
                    )

        if mutual_relations:
            lines.append("\nKey relationships:")
            lines.extend(mutual_relations[:3])

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Graph update
    # ------------------------------------------------------------------

    def update(self, all_players, current_round: int, discussion_history: str, model_name: str, bigfive_estimates: dict = None, night_outcome: str = "") -> None:
        """
        Update the trust graph based on the last round's discussion using an LLM call.

        Args:
            all_players (list): List of all Player objects in the game.
            current_round (int): Current round number.
            discussion_history (str): The last round's discussion text (already stripped of think tags).
            model_name (str): The LLM model name to use for the update call.
            bigfive_estimates (dict, optional): The owning player's latest Big Five estimates
                for other players, mapping player_name -> BigFiveProfile (or dict with scores).
            night_outcome (str, optional): A description of what happened during the
                just‑completed night phase (e.g. “X was killed by the Mafia.”
                or “The Doctor protected Y.”). Will be included in the LLM prompt.
        """
        if self.graph is None:
            return

        if not discussion_history or discussion_history.strip() == "":
            return

        # Update alive status in graph (nodes keyed by player_name)
        for player in all_players:
            if player.player_name in self.graph.nodes:
                self.graph.nodes[player.player_name]["alive"] = player.alive

        alive_players = [p for p in all_players if p.alive]

        # Skip if only self alive
        if len(alive_players) <= 1:
            return

        prompt = self._build_update_prompt(alive_players, discussion_history, current_round, bigfive_estimates, night_outcome)
        if prompt is None:
            return

        MAX_ATTEMPTS = 2
        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                response = get_llm_response(model_name, prompt)
                updates = self._parse_updates(response)
                self._apply_updates(updates, alive_players, current_round)
                return  # success – updates applied
            except (json.JSONDecodeError, ValueError, Exception) as e:
                if attempt == MAX_ATTEMPTS:
                    print(f"[{self.player_name}] Graph update failed after {MAX_ATTEMPTS} attempts: {e}")
                else:
                    print(f"[{self.player_name}] Graph update attempt {attempt} failed, retrying: {e}")

    def _parse_updates(self, response: str) -> dict:
        """
        Extract the JSON payload from the LLM response and parse it.

        Raises ValueError if no JSON is found; raises JSONDecodeError on parse failure.
        """
        json_match = re.search(r"\{[\s\S]*\}", response)
        if not json_match:
            raise ValueError("No JSON found in graph update response")
        return json.loads(json_match.group())

    def _build_update_prompt(self, alive_players, last_round_history: str, current_round: int, bigfive_estimates: dict = None, night_outcome: str = ""):
        """
        Build the prompt for the LLM to evaluate trust and role probabilities.
        All players are referenced by player_name.

        Args:
            alive_players (list): List of alive Player objects.
            last_round_history (str): The last round's discussion text.
            current_round (int): Current round number.
            bigfive_estimates (dict, optional): Latest Big Five estimates for other players,
                keyed by player_name.
            night_outcome (str, optional): What happened during the night phase (if any).

        Returns:
            str or None: The prompt string, or None if there is nothing to evaluate.
        """
        graph_state = self.to_prompt(alive_players)

        if self.role == Role.MAFIA:
            mafia_members = [
                p.player_name
                for p in alive_players
                if self.graph.nodes[p.player_name]["role_probabilities"].get("Mafia", 0) == 1.0
                and p.player_name != self.player_name
            ]
            role_context = (
                f"You are MAFIA. Your goal is to eliminate villagers without being detected.\n"
                f"Known mafia allies: {', '.join(mafia_members) if mafia_members else 'None (you are alone)'}\n"
                f"You should evaluate who is suspicious of you and who might be easy targets."
            )
        elif self.role == Role.DOCTOR:
            role_context = (
                "You are DOCTOR. Your goal is to identify mafia and protect key villagers.\n"
                "Pay attention to who seems to be leading the town and who might be targeted."
            )
        else:
            role_context = (
                "You are VILLAGER. Your goal is to identify and eliminate mafia members.\n"
                "Look for suspicious behavior, inconsistencies, and unusual voting patterns."
            )

        # Players to evaluate (exclude self; for mafia — exclude known allies)
        players_to_evaluate = []
        for p in alive_players:
            if p.player_name == self.player_name:
                continue
            if self.role == Role.MAFIA:
                probs = self.graph.nodes[p.player_name]["role_probabilities"]
                if probs.get("Mafia", 0) == 1.0:
                    continue
            players_to_evaluate.append(p.player_name)

        if not players_to_evaluate:
            return None

        # ----- Big Five estimates section (if available) -----
        bigfive_section = ""
        if bigfive_estimates:
            bf_lines = []
            for pname in players_to_evaluate:
                profile = bigfive_estimates.get(pname)
                if profile is not None:
                    # profile may be a BigFiveProfile dataclass or a plain dict
                    if hasattr(profile, 'openness'):
                        o = profile.openness
                        c = profile.conscientiousness
                        e = profile.extraversion
                        a = profile.agreeableness
                        n = profile.neuroticism
                    else:
                        o = profile.get('openness', 3.0)
                        c = profile.get('conscientiousness', 3.0)
                        e = profile.get('extraversion', 3.0)
                        a = profile.get('agreeableness', 3.0)
                        n = profile.get('neuroticism', 3.0)
                    bf_lines.append(
                        f"  - {pname}: Openness={o:.1f}, Conscientiousness={c:.1f}, "
                        f"Extraversion={e:.1f}, Agreeableness={a:.1f}, Neuroticism={n:.1f}"
                    )
            if bf_lines:
                bigfive_section = (
                    "\n\nYOUR CURRENT BIG FIVE ESTIMATES FOR OTHER PLAYERS "
                    "(based on their messages so far):\n" + "\n".join(bf_lines)
                )

        # ----- Night outcome section (if available) -----
        night_outcome_section = ""
        if night_outcome:
            night_outcome_section = f"\n\n=== LAST NIGHT'S OUTCOME ===\n{night_outcome}"

        prompt = f"""You are {self.player_name} analyzing Round {current_round} of a Mafia game.

{role_context}

=== LAST ROUND'S DISCUSSION ===
{last_round_history}

=== YOUR CURRENT ASSESSMENTS ===
{graph_state}{bigfive_section}{night_outcome_section}

=== TASK ===
Based on the discussion, update your assessment of these players: {', '.join(players_to_evaluate)}

For each player, evaluate:
1. **mafia_probability**: How likely they are Mafia (0.0 = definitely innocent, 1.0 = definitely Mafia)
2. **trust**: Your trust in them (-1.0 = complete distrust, 0.0 = neutral, 1.0 = complete trust)

Also note any trust/distrust you observed BETWEEN other players.

Consider:
- Who accused whom and how convincingly?
- Who defended whom?
- Voting patterns and flip-flopping
- Emotional reactions vs logical arguments
- Who tries to lead vs who stays quiet?
- (If Big Five estimates are available) how consistent a player's traits are with a typical Mafia or innocent profile.

Respond ONLY with valid JSON:
{{
    "player_assessments": {{
        "PlayerName": {{
            "mafia_probability": 0.3,
            "trust": 0.2,
            "reasoning": "brief explanation"
        }}
    }},
    "observed_relationships": [
        {{
            "from": "Player1",
            "to": "Player2",
            "trust_change": 0.3,
            "reasoning": "Player1 defended Player2"
        }}
    ]
}}
"""
        return prompt

    def _apply_updates(self, updates: dict, alive_players, current_round: int) -> None:
        """
        Parse the LLM response and apply updates to the graph using incremental blending.
        All graph nodes are keyed by player_name.

        Args:
            updates (dict): Already parsed JSON containing player_assessments and observed_relationships.
            alive_players (list): List of alive Player objects.
            current_round (int): Current round number.
        """
        alive_names = {p.player_name for p in alive_players}

        # Blending factor: how much weight to give new assessment vs old
        BLEND_FACTOR = 0.6

        # Apply player assessments (keyed by player_name)
        if "player_assessments" in updates:
            for player_name, assessment in updates["player_assessments"].items():
                if player_name not in self.graph.nodes:
                    continue
                if player_name == self.player_name:
                    continue
                if player_name not in alive_names:
                    continue

                # Skip known mafia allies for mafia players
                if self.role == Role.MAFIA:
                    current_probs = self.graph.nodes[player_name]["role_probabilities"]
                    if current_probs.get("Mafia", 0) == 1.0:
                        continue

                # Update role probabilities
                if "mafia_probability" in assessment:
                    try:
                        new_mafia_prob = float(assessment["mafia_probability"])
                        new_mafia_prob = max(0.0, min(1.0, new_mafia_prob))

                        old_probs = self.graph.nodes[player_name]["role_probabilities"]
                        old_mafia_prob = old_probs.get("Mafia", 0.5)
                        blended_mafia = (
                            old_mafia_prob * (1 - BLEND_FACTOR) + new_mafia_prob * BLEND_FACTOR
                        )

                        remaining = 1.0 - blended_mafia
                        self.graph.nodes[player_name]["role_probabilities"] = {
                            "Mafia": blended_mafia,
                            "Villager": remaining * 0.75,
                            "Doctor": remaining * 0.25,
                        }
                    except (ValueError, TypeError):
                        pass

                # Update trust edge from self to this player (by player_name)
                if "trust" in assessment:
                    try:
                        new_trust = float(assessment["trust"])
                        new_trust = max(-1.0, min(1.0, new_trust))

                        old_trust = self.graph[self.player_name][player_name].get("trust", 0.0)
                        blended_trust = (
                            old_trust * (1 - BLEND_FACTOR) + new_trust * BLEND_FACTOR
                        )

                        self.graph[self.player_name][player_name]["trust"] = blended_trust
                        self.graph[self.player_name][player_name]["last_updated"] = current_round

                        if "reasoning" in assessment:
                            evidence = self.graph[self.player_name][player_name].get(
                                "evidence", []
                            )
                            evidence.append(
                                {
                                    "round": current_round,
                                    "reason": assessment["reasoning"][:100],
                                }
                            )
                            self.graph[self.player_name][player_name]["evidence"] = evidence[-5:]

                    except (ValueError, TypeError):
                        pass

        # Apply observed relationships between other players (by player_name)
        if "observed_relationships" in updates:
            for rel in updates["observed_relationships"]:
                try:
                    from_player = rel.get("from", "")
                    to_player = rel.get("to", "")
                    trust_change = float(rel.get("trust_change", 0))

                    if from_player not in alive_names or to_player not in alive_names:
                        continue
                    if from_player == self.player_name:
                        continue  # Already handled above
                    if from_player == to_player:
                        continue
                    if from_player not in self.graph or to_player not in self.graph[from_player]:
                        continue

                    trust_change = max(-0.5, min(0.5, trust_change))
                    old_trust = self.graph[from_player][to_player].get("trust", 0.0)
                    new_trust = max(-1.0, min(1.0, old_trust + trust_change))

                    self.graph[from_player][to_player]["trust"] = new_trust
                    self.graph[from_player][to_player]["last_updated"] = current_round

                except (ValueError, TypeError, KeyError):
                    continue
