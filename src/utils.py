import numpy as np

type Vec = np.ndarray[tuple[int], np.dtype[np.float64]]
type Mat = np.ndarray[tuple[int, int], np.dtype[np.float64]]
type Observation = np.ndarray[tuple[int], np.dtype[np.uint8]]
type RMState = int
type State = tuple[Observation, RMState]
type Action = int
type Reward = float
type Experience = tuple[State, State, float, bool]
type Experiences = list[Experience]
type Props = dict[str, bool]


def hash_state(state: State) -> tuple[int, ...]:
    obs, u = state
    return (u, *tuple(obs.astype(int).tolist()))
