import config
import re
import json
from openrouter import get_llm_response


class GameOrchestrator:
    """Handles high‑level game orchestration: graphs, full game loop, critic review."""

    def __init__(self, game):
        """
        Args:
            game: MafiaGame instance that provides players, state, logger, etc.
        """
        self.game = game

    def init_all_graphs(self):
        """Initialize relationship graphs for all players."""
        for player in self.game.players:
            if player.use_graph:
                player.init_graph(self.game.players)
                print(f"[Graph] Initialized graph for {player.player_name}")

    def update_all_graphs(self, current_round):
        """Update relationship graphs for alive graph‑using players."""
        all_players = self.game.players
        for player in self.game.get_alive_players():
            if player.alive and player.use_graph:
                try:
                    player.update_graph(all_players, current_round)
                    print(f"[Graph] Updated graph for {player.player_name}")
                except Exception as e:
                    print(f"[Graph] Failed to update graph for {player.player_name}: {e}")

    def run_game(self):
        """
        Run the complete game loop.

        Returns:
            tuple: (winner, rounds_data, participants, language, critic_review)
        """
        if not self.game.setup_game():
            return None, [], {}, self.game.language

        game_over = False
        winner = None

        while not game_over:
            game_over, winner = self.game.check_game_over()
            if game_over:
                break

            print(f'round_number={self.game.round_number}')
            if self.game.round_number == 1:
                self.init_all_graphs()

            # Execute day phase
            self.game.execute_day_phase()

            # Check if game is over after day
            game_over, winner = self.game.check_game_over()
            if game_over:
                break

            # Execute night phase
            self.game.execute_night_phase()

            self.update_all_graphs(self.game.round_number)

        # Add final round data if not already added
        if self.game.current_round_data["round_number"] > 0:
            self.game.state.add_round_data(self.game.current_round_data)

        # Create participants dictionary keyed by player_name
        participants = {}
        for player in self.game.players:
            participants[player.player_name] = {
                "role": player.role.value,
                "model_name": player.model_name,
                "player_name": player.player_name,
            }

        # Generate game critic review
        critic_review = self.generate_critic_review(winner)

        # Log game end
        self.game.logger.game_end(1, winner, self.game.round_number)

        return winner, self.game.rounds_data, participants, self.game.language, critic_review

    def generate_critic_review(self, winner):
        """
        Generate a game critic review using Claude via OpenRouter.

        Args:
            winner (str): The winning team ("Mafia" or "Villagers").

        Returns:
            dict: A dictionary with title, content, and one_liner.
        """
        game_summary = {
            "winner": winner,
            "rounds": self.game.round_number,
            "participants": {
                player.player_name: player.role.value for player in self.game.players
            },
            "eliminations": [],
        }

        for round_data in self.game.rounds_data:
            if "eliminations" in round_data and round_data["eliminations"]:
                for player_name in round_data["eliminations"]:
                    game_summary["eliminations"].append(
                        {
                            "player": player_name,
                            "round": round_data["round_number"],
                            "phase": round_data.get("phase", "unknown"),
                        }
                    )

        prompt = f"""You are a professional game critic reviewing a Mafia game played by AI language models. 
        
Game summary:
- Winner: {winner}
- Number of rounds: {self.game.round_number}
- Players and roles: {game_summary['participants']}
- Eliminations: {game_summary['eliminations']}

Write a short, entertaining critic review of this game. Include:
1. A catchy title for your review (max 50 characters)
2. A concise review (max 200 words) that analyzes:
   - The game's pacing and length
   - Interesting strategic moves or blunders
   - The performance of the winning team
   - Any particularly noteworthy moments
3. A one-sentence intense summary that captures the essence of the game in a dramatic way (max 100 characters)

Your tone should be professional but entertaining, like a game critic. Be specific about this particular game.
Format your response as a JSON object with 'title', 'content', and 'one_liner' fields.
"""

        try:
            model_name = config.CLAUDE_SONNET_4
            response_content = get_llm_response(model_name, prompt)

            if response_content == "ERROR: Could not get response":
                return {
                    "title": "Game Review Unavailable",
                    "content": "The critic was unable to review this game due to API issues.",
                    "one_liner": "Technical difficulties prevented our critic from witnessing this showdown.",
                }

            json_match = re.search(r"({.*})", response_content, re.DOTALL)

            if json_match:
                try:
                    review_json = json.loads(json_match.group(1))
                    if "one_liner" not in review_json:
                        review_json["one_liner"] = (
                            "A game that defies simple description!"
                        )
                    return review_json
                except json.JSONDecodeError:
                    return {
                        "title": "AI Mafia Game Review",
                        "content": response_content[:300],
                        "one_liner": "A game that left our critic speechless!",
                    }
            else:
                return {
                    "title": "AI Mafia Game Review",
                    "content": response_content[:300],
                    "one_liner": "A game that defies conventional criticism!",
                }

        except Exception as e:
            print(f"Error generating critic review: {e}")
            return {
                "title": "Game Review Unavailable",
                "content": "The critic was unable to review this game due to technical difficulties.",
                "one_liner": "Technical issues prevented our critic from delivering judgment.",
            }
