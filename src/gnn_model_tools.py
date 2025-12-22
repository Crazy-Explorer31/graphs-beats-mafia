import torch
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GATv2Conv, global_mean_pool
import networkx as nx
import numpy as np
import json
import os

ROLE_MAP = {"Villager": 0, "Doctor": 1, "Mafia": 2}
INV_ROLE_MAP = {0: "Villager", 1: "Doctor", 2: "Mafia"}

def encode_role_probs(prob_dict):
    """Гарантируем порядок: [Villager, Doctor, Mafia]"""
    return [
        prob_dict.get("Villager", 0.0),
        prob_dict.get("Doctor", 0.0),
        prob_dict.get("Mafia", 0.0)
    ]

def graph_to_data(G, full_ground_truth=None):
    """
    Преобразует граф NetworkX в тензоры PyG.
    full_ground_truth: dict {player_name: role_str} (нужен только для обучения)
    """
    nodes = list(G.nodes())
    node_to_idx = {name: i for i, name in enumerate(nodes)}

    x_features = []
    y_labels = []
    train_mask = []

    for node_name in nodes:
        attrs = G.nodes[node_name]

        # 1. Формируем вектор признаков (5 значений)
        # [P_Villager, P_Doctor, P_Mafia, Alive, Is_Self]
        probs = encode_role_probs(attrs['role_probabilities'])
        is_alive = 1.0 if attrs['alive'] else 0.0
        is_self = 1.0 if attrs.get('is_self', False) else 0.0

        x_features.append(probs + [is_alive, is_self])

        # 2. Формируем метки для обучения
        if full_ground_truth:
            true_role = full_ground_truth.get(node_name, "Villager")
            y_labels.append(ROLE_MAP[true_role])
            # Не учимся на самом себе (мы и так знаем свою роль)
            train_mask.append(not attrs.get('is_self', False))
        else:
            y_labels.append(0) # Заглушка
            train_mask.append(False)

    # 3. Ребра
    edge_indices = []
    edge_attrs = []

    for u, v, data in G.edges(data=True):
        src = node_to_idx[u]
        dst = node_to_idx[v]
        edge_indices.append([src, dst])
        # Признак ребра: trust (-1..1)
        edge_attrs.append([data.get('trust', 0.0)])

    # 4. Сборка объекта Data
    x = torch.tensor(x_features, dtype=torch.float)
    y = torch.tensor(y_labels, dtype=torch.long)
    edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attrs, dtype=torch.float)
    train_mask = torch.tensor(train_mask, dtype=torch.bool)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, train_mask=train_mask)

class MafiaGAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        # edge_dim=1, так как 1 параметр (trust)
        self.conv1 = GATv2Conv(in_channels, hidden_channels, heads=4, concat=True, edge_dim=1)
        # Вход следующего слоя = hidden * heads
        self.conv2 = GATv2Conv(hidden_channels * 4, hidden_channels, heads=2, concat=False, edge_dim=1)
        self.classifier = torch.nn.Linear(hidden_channels, out_channels)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        x = self.conv1(x, edge_index, edge_attr=edge_attr)
        x = F.elu(x)
        x = F.dropout(x, p=0.4, training=self.training)

        x = self.conv2(x, edge_index, edge_attr=edge_attr)
        x = F.elu(x)

        return self.classifier(x)

def predict_game_roles(model, agent_graph):
    """
    Функция для использования в продакшене (в игре).
    Принимает граф networkx, возвращает dict с вероятностями.
    """
    model.eval()

    # Конвертируем граф (ground_truth=None, так как мы его не знаем)
    data = graph_to_data(agent_graph, full_ground_truth=None)

    with torch.no_grad():
        # Получаем логиты
        logits = model(data)
        # Превращаем в вероятности (Softmax)
        probs_tensor = F.softmax(logits, dim=1)

    # Форматируем вывод
    result = {}
    nodes = list(agent_graph.nodes())

    for i, node_name in enumerate(nodes):
        # probs_tensor[i] -> [p_villager, p_doctor, p_mafia]
        p = probs_tensor[i].tolist()
        result[node_name] = {
            "Villager": round(p[0], 3),
            "Doctor":   round(p[1], 3),
            "Mafia":    round(p[2], 3)
        }

    return result