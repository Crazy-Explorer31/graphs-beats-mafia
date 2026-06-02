import re
import uuid
import config

class GameStateManager:
    def __init__(self, language):
        self.game_id = str(uuid.uuid4())
        self.players = []
        self.mafia_players = []
        self.doctor_player = None
        self.villager_players = []
        self.round_number = 0
        self.phase = "setup"
        self.language = language
        self.rounds_data = []
        self.current_round_data = {
            "round_number": 0,
            "messages": [],
            "actions": {},
            "eliminations": [],
            "eliminated_by_vote": [],
            "targeted_by_mafia": [],
            "protected_by_doctor": [],
            "outcome": "",
        }

    def set_players(self, players, mafia_players, doctor_player, villager_players):
        self.players = players
        self.mafia_players = mafia_players
        self.doctor_player = doctor_player
        self.villager_players = villager_players

    def get_alive_players(self):
        return [p for p in self.players if p.alive]

    def get_game_state(self):
        alive_count = sum(1 for p in self.players if p.alive)
        mafia_count = sum(1 for p in self.mafia_players if p.alive)
        villager_count = sum(1 for p in self.villager_players if p.alive)
        doctor_count = 1 if self.doctor_player and self.doctor_player.alive else 0

        state = f"Round {self.round_number}, {self.phase.capitalize()} phase. "
        state += f"{alive_count} players alive ({mafia_count} Mafia, {villager_count + doctor_count} Villagers/Doctor). "

        if self.round_number > 1:
            eliminations = self.current_round_data.get("eliminations", [])
            if eliminations:
                state += f"In the previous round, {', '.join(eliminations)} {'was' if len(eliminations) == 1 else 'were'} eliminated. "

        return state

    def check_game_over(self):
        mafia_alive = sum(1 for p in self.mafia_players if p.alive)
        villagers_alive = sum(1 for p in self.villager_players if p.alive)
        doctor_alive = 1 if self.doctor_player and self.doctor_player.alive else 0

        if mafia_alive == 0:
            return True, "Villagers"
        elif mafia_alive >= (villagers_alive + doctor_alive):
            return True, "Mafia"
        elif self.round_number >= config.MAX_ROUNDS:
            if villagers_alive + doctor_alive > mafia_alive:
                return True, "Villagers"
            else:
                return True, "Mafia"

        return False, None

    def add_round_data(self, round_data):
        self.rounds_data.append(round_data)


class DiscussionHistory:
    def __init__(self):
        self.history = ""
        self.last_round_history = ""

    @staticmethod
    def without_thinking(text):
        # Remove properly closed think tags
        text = re.sub(
            r"<[tT][hH][iI][nN][kK]>.*?</[tT][hH][iI][nN][kK]>",
            "",
            text,
            flags=re.DOTALL,
        )
        # Remove unclosed think tags
        text = re.sub(
            r"<[tT][hH][iI][nN][kK]>.*$",
            "",
            text,
            flags=re.DOTALL,
        )
        return text

    def discussion_history_without_thinking(self):
        return self.without_thinking(self.history)

    def discussion_history_last_round_without_thinking(self):
        return self.without_thinking(self.last_round_history)
