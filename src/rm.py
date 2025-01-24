import re
from collections import defaultdict
from dataclasses import dataclass

State = tuple[int, int]
Reward = float
Terminal = bool
Props = dict[str, bool]
CRM_Info = dict[State, tuple[State, Reward, Terminal]]


def extract_variables(formula: str) -> set[str]:
    return set(re.findall(r"\b[a-zA-Z_]\w*\b", formula))


def parse_formula(formula: str) -> str:
    formula = re.sub(r"(?<!\w)!(\w+)", r" not \1", formula)
    formula = re.sub(r"&", " and ", formula)
    formula = re.sub(r"\|", " or ", formula)
    return formula


def evaluate_formula(formula: str, variables: Props) -> bool:
    try:
        # WARN: can execute arbitrary code
        # TODO: make custom evaluator
        return bool(eval(formula, {}, variables))
    except Exception as e:
        raise ValueError(f"Invalid formula: '{formula}'. Error: {e}")


@dataclass
class RM:
    # https://github.com/RodrigoToroIcarte/reward_machines/blob/master/reward_machines/reward_machines/reward_machine.py

    def __init__(self, transitions: list[tuple[int, int, str, float]]) -> None:
        assert len(transitions) != 0, "Expected non-empty transition list."

        states_set: set[int] = set()
        terminals_set: set[int] = set()
        variables: set[str] = set()
        self.delta_u: dict[int, dict[int, str]] = defaultdict(dict)
        self.delta_r: dict[int, dict[int, float]] = defaultdict(dict)

        for u1, u2, formula, reward in transitions:
            states_set.add(u1)
            terminals_set.add(u2)
            variables.update(extract_variables(formula))
            self.delta_u[u1][u2] = parse_formula(formula)
            self.delta_r[u1][u2] = reward

        terminals_set.difference_update(states_set)

        self.states: list[int] = sorted(states_set)
        self.initial_state: int = self.states[0]
        self.terminal_states: list[int] = sorted(terminals_set)
        self.variables: list[str] = sorted(variables)
        self.transitions: dict[tuple[int, int], int] = dict()

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

    def get_states(self) -> list[int]:
        return self.states

    def get_bitmask(self, props: Props) -> int:
        bitmask = 0
        for i, (prop, val) in enumerate(props.items()):
            if val and prop in self.variables:
                bitmask |= 1 << i
        return bitmask

    def next_state(self, u1: int, props: Props) -> int:
        bitmask = self.get_bitmask(props)
        u2 = self.transitions.get((u1, bitmask))
        if u2 is None:
            u2 = self.compute_next_state(u1, props)
            self.transitions[(u1, bitmask)] = u2
        return u2

    def compute_next_state(self, u1: int, props: Props) -> int:
        for u2 in self.delta_u[u1]:
            if evaluate_formula(self.delta_u[u1][u2], props):
                return u2
        raise ValueError(
            f"No transition found for state {u1} and propositions {props}."
        )

    def get_reward(self, u1: int, u2: int) -> float:
        return self.delta_r[u1][u2]

    def get_crm(self, s1: int, s2: int, props: Props) -> CRM_Info:
        crm_info: CRM_Info = dict()
        for u1 in self.states:
            u2 = self.next_state(u1, props)
            r = self.get_reward(u1, u2)
            done = u2 in self.terminal_states
            crm_info[(s1, u1)] = ((s2, u2), r, done)
        return crm_info

    def step(
        self,
        state: tuple[int, int],
        next_env_state: int,
        props: Props,
    ) -> tuple[State, float, bool, CRM_Info]:
        s1, u1 = state
        assert u1 not in self.terminal_states, "Expected non-terminal state."
        u2 = self.next_state(u1, props)
        next_state = (next_env_state, u2)
        reward = self.get_reward(u1, u2)
        done = u2 in self.terminal_states
        crm_info = self.get_crm(s1, next_env_state, props)
        return next_state, reward, done, crm_info
