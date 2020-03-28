"""
File Name:          faker.py
Project:            biochem-graph

File Description:

"""
from typing import Optional
from random import randint


class Faker:
    def __init__(
            self,
            random_state: Optional[int] = None,
    ):
        if random_state:
            self.__random_state = random_state

    def mol(self):
        pass

    def mol_smiles(self):
        pass

    def mol_graph(self):
        pass