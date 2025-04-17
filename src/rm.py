import re
from collections import defaultdict
from dataclasses import dataclass

State = tuple[int, int]
Experience = dict[State, tuple[State, float, bool]]


def extract_variables(formula: str) -> set[str]:
    return set(re.findall(r"\b[a-zA-Z_]\w*\b", formula))


def parse_formula(formula: str) -> str:
    formula = re.sub(r"(?<!\w)!(\w+)", r" not \1", formula)
    formula = re.sub(r"&", " and ", formula)
    formula = re.sub(r"\|", " or ", formula)
    return formula


def evaluate_formula(formula: str, props: dict[str, bool]) -> bool:
    try:
        # WARN: can execute arbitrary code
        # TODO: make custom evaluator
        return bool(eval(formula, {}, props))
    except Exception as e:
        raise ValueError(f"Invalid formula: '{formula}'. Error: {e}")


@dataclass
class RM:
    # https://github.com/RodrigoToroIcarte/reward_machines/blob/master/reward_machines/reward_machines/reward_machine.py

    def __init__(self, transitions: list[tuple[int, int, str, float]]) -> None:
        assert len(transitions) != 0, "Expected non-empty transition list."

        states_set: set[int] = set()
        terminals_set: set[int] = set()
        self.delta_u: dict[int, dict[int, str]] = defaultdict(dict)
        self.delta_r: dict[int, dict[int, float]] = defaultdict(dict)

        for u1, u2, formula, reward in transitions:
            states_set.add(u1)
            terminals_set.add(u2)
            self.delta_u[u1][u2] = parse_formula(formula)
            self.delta_r[u1][u2] = reward

        terminals_set.difference_update(states_set)

        self.states: list[int] = sorted(states_set)
        self.initial_state: int = self.states[0]
        self.terminal_states: list[int] = sorted(terminals_set)

    @staticmethod
    def from_file(filename: str) -> "RM":
        with open(filename, "r") as file:
            lines = file.readlines()

        # WARN: can execute arbitrary code
        # TODO: create custom DSL and parser
        transitions: list[tuple[int, int, str, float]] = [eval(line) for line in lines]

        return RM(transitions)

    def reset(self) -> int:
        return self.initial_state

    def get_next_state(self, u1: int, props: dict[str, bool]) -> int:
        for u2 in self.delta_u[u1]:
            if evaluate_formula(self.delta_u[u1][u2], props):
                return u2
        raise ValueError(
            f"No transition found for state {u1} and propositions {props}."
        )

    def get_reward(self, u1: int, u2: int) -> float:
        return self.delta_r[u1][u2]

    def get_experience(self, s1: int, s2: int, props: dict[str, bool]) -> Experience:
        experience: Experience = dict()
        for u1 in self.states:
            u2 = self.get_next_state(u1, props)
            r = self.get_reward(u1, u2)
            terminal = u2 in self.terminal_states
            experience[(s1, u1)] = ((s2, u2), r, terminal)
        return experience

    def step(
        self,
        state: State,
        next_env_state: int,
        props: dict[str, bool],
    ) -> tuple[State, float, bool, Experience]:
        s1, u1 = state
        assert u1 not in self.terminal_states, "Expected non-terminal state."
        experience = self.get_experience(s1, next_env_state, props)
        next_state, reward, terminal = experience[state]
        return next_state, reward, terminal, experience
