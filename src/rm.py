import re
from collections import defaultdict
from dataclasses import dataclass

from utils import Experiences, Observation, Props, Reward, RMState, State


def extract_variables(formula: str) -> set[str]:
    return set(re.findall(r"\b[a-zA-Z_]\w*\b", formula))


def parse_formula(formula: str) -> str:
    formula = re.sub(r"(?<!\w)!(\w+)", r" not \1", formula)
    formula = re.sub(r"&", " and ", formula)
    formula = re.sub(r"\|", " or ", formula)
    return formula


def evaluate_formula(formula: str, props: Props) -> bool:
    try:
        # WARN: can execute arbitrary code
        # TODO: make custom evaluator
        return bool(eval(formula, {}, props))
    except Exception as e:
        raise ValueError(f"Invalid formula: '{formula}'. Error: {e}")


@dataclass
class RM:
    # https://github.com/RodrigoToroIcarte/reward_machines/blob/master/reward_machines/reward_machines/reward_machine.py

    def __init__(self, transitions: list[tuple[RMState, RMState, str, Reward]]) -> None:
        assert len(transitions) != 0, "Expected non-empty transition list."

        states_set: set[RMState] = set()
        terminals_set: set[RMState] = set()
        self.delta_u: dict[RMState, dict[RMState, str]] = defaultdict(dict)
        self.delta_r: dict[RMState, dict[RMState, Reward]] = defaultdict(dict)

        for u1, u2, formula, reward in transitions:
            states_set.add(u1)
            terminals_set.add(u2)
            self.delta_u[u1][u2] = parse_formula(formula)
            self.delta_r[u1][u2] = reward

        terminals_set.difference_update(states_set)

        self.states: list[RMState] = sorted(states_set)
        self.initial_state: RMState = self.states[0]
        self.terminal_states: list[RMState] = sorted(terminals_set)

    @staticmethod
    def from_file(filename: str) -> "RM":
        with open(filename, "r") as file:
            lines = file.readlines()

        # WARN: can execute arbitrary code
        # TODO: create custom DSL and parser
        transitions: list[tuple[RMState, RMState, str, Reward]] = [
            eval(line) for line in lines
        ]

        return RM(transitions)

    def reset(self) -> RMState:
        return self.initial_state

    def get_next_state(self, u1: RMState, props: Props) -> RMState:
        for u2 in self.delta_u[u1]:
            if evaluate_formula(self.delta_u[u1][u2], props):
                return u2
        raise ValueError(
            f"No transition found for state {u1} and propositions {props}."
        )

    def get_reward(self, u1: RMState, u2: RMState) -> Reward:
        return self.delta_r[u1][u2]

    def get_experiences(
        self, s1: Observation, s2: Observation, props: Props
    ) -> Experiences:
        experiences: Experiences = []
        for u1 in self.states:
            u2 = self.get_next_state(u1, props)
            r = self.get_reward(u1, u2)
            terminal = u2 in self.terminal_states
            experiences.append(((s1, u1), (s2, u2), r, terminal))
        return experiences

    def step(
        self,
        state: State,
        obs: Observation,
        props: Props,
    ) -> tuple[State, Reward, bool, Experiences]:
        s1, u1 = state
        assert u1 not in self.terminal_states, "Expected non-terminal state."
        u2 = self.get_next_state(u1, props)
        reward = self.get_reward(u1, u2)
        terminal = u2 in self.terminal_states
        experiences = self.get_experiences(s1, obs, props)
        return (obs, u2), reward, terminal, experiences
