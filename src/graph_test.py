from player import Player
from game_templates import Role

villager1 = Player(model_name="Test1", player_name="Test1", role=Role.VILLAGER, use_graph=True)
villager2 = Player(model_name="Test2", player_name="Test2", role=Role.VILLAGER, use_graph=True)
villager3 = Player(model_name="Test3", player_name="Test3", role=Role.VILLAGER, use_graph=True)
doctor = Player(model_name="Test4", player_name="Test4", role=Role.DOCTOR, use_graph=True)
mafia1 = Player(model_name="Test5", player_name="Test5", role=Role.MAFIA, use_graph=True)
mafia2 = Player(model_name="Test6", player_name="Test6", role=Role.MAFIA, use_graph=True)

all_players = [
    villager1,
    villager2,
    villager3,
    doctor,
    mafia1,
    mafia2
]

g_villager = villager1.init_graph(all_players)
print(g_villager)

p_villager = villager1.graph_to_prompt(all_players)
print(p_villager)

g_doctor = doctor.init_graph(all_players)
print(g_doctor)

p_doctor = doctor.graph_to_prompt(all_players)
print(p_doctor)

g_mafia = mafia1.init_graph(all_players)
print(g_mafia)

p_mafia1 = mafia1.graph_to_prompt(all_players)
print(p_mafia1)
