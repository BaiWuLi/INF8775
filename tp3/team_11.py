import numpy as np
import random
import time
from typing import List, Tuple, Dict


class SpongeGridProblem:
    """
    Représente une instance du problème.

    - n: dimension du carré n×n à produire
    - alphabet: liste de symboles permis (ex: ["A","B","C","I"])
    - targets: séquences qu'on veut VOIR dans la grille
    - banned: séquences qu'on veut ÉVITER dans la grille
    """
    def __init__(
        self,
        n: int,
        alphabet: List[str],
        targets: List[str],
        banned: List[str]
    ):
        self.n = n
        self.alphabet = alphabet
        self.targets = targets
        self.banned = banned

def sequences_in_line(line: str, seq: str) -> bool:
    """True si 'seq' apparaît comme sous-chaîne contiguë dans 'line'."""
    return seq in line


def count_matches_in_grid(grid: List[List[str]], seqs: List[str]) -> int:
    """
    Combien de séquences DISTINCTES dans `seqs` apparaissent au moins une fois
    dans le carré (en ligne gauche->droite ou en colonne haut->bas) ?
    """
    n = len(grid)
    # Prépare toutes les lignes (strings)
    rows = ["".join(grid[i][j] for j in range(n)) for i in range(n)]
    cols = ["".join(grid[i][j] for i in range(n)) for j in range(n)]

    found = 0
    for s in seqs:
        present = any(sequences_in_line(r, s) for r in rows) \
                  or any(sequences_in_line(c, s) for c in cols)
        if present:
            found += 1
    return found


def score_grid(
    grid: List[List[str]],
    problem: SpongeGridProblem,
    alpha: float = 1.0,
    beta: float = 1.0
) -> Dict[str, float]:
    """
    Calcule le score = alpha * (#objectifs couverts) - beta * (#bannies déclenchées)
    et renvoie aussi les sous-métriques.
    """
    covered_obj = count_matches_in_grid(grid, problem.targets)
    triggered_bad = count_matches_in_grid(grid, problem.banned)
    score_val = alpha * covered_obj - beta * triggered_bad
    return {
        "score": score_val,
        "covered": covered_obj,
        "triggered": triggered_bad
    }

import numpy as np
from time import perf_counter

def naive_solution(problem: SpongeGridProblem):
    n = problem.n
    targets: list[str] = random.sample(problem.targets, k=len(problem.targets)) # shuffle targets
    alphabet: list[str] = problem.alphabet # list[char]

    targets = [char for string in targets for char in string]
    if len(targets) < n*n:
        # fill with random alphabet characters
        targets += [random.choice(alphabet) for _ in range(n*n - len(targets))]
    else:
        targets = targets[:n*n]

    grid = [targets[i*n:(i+1)*n] for i in range(n)] # reshape into n x n grid

    return np.array(grid)

def rand_change(n: int, alphabet: list[str]) -> Tuple[Tuple[int, int], str]:
    cell = (random.randint(0, n-1), random.randint(0, n-1))
    char = random.choice(alphabet)

    return cell, char

def eval_change(grid: np.ndarray, cell: Tuple[int, int], char: str, problem: SpongeGridProblem):
    orig_char = grid[cell]
    orig_score = score_grid(grid, problem)["score"]
    grid[cell] = char
    new_score = score_grid(grid, problem)["score"]
    grid[cell] = orig_char

    return new_score - orig_score

def simulated_annealing(problem: SpongeGridProblem, max_time: float = 180.0, temp: float = 100000.0, min_temp: float = 1e-6, cooling_rate: float = 0.9999):
    start_time = perf_counter()
    best_grid = naive_solution(problem)
    best_score = score_grid(best_grid, problem)["score"]

    grid = best_grid.copy()
    score = best_score
    # i = 1000 # 1000 itérations, utilisées pour la partie analyse hybride
    
    while temp > min_temp and (perf_counter() - start_time) < max_time:
    # while i > 0:
    #     i -= 1
        cell, char = rand_change(problem.n, problem.alphabet)
        delta = eval_change(grid, cell, char, problem)

        if delta > 0 or random.uniform(0, 1) < np.exp(delta / temp):
            grid[cell] = char
            score += delta

            if score >= best_score:
                best_grid = grid.copy()
                best_score = score
                
        temp *= cooling_rate

    return best_grid

from ortools.sat.python import cp_model

def add_sequence_in_line_constraint(
    model: cp_model.CpModel,
    line: List[cp_model.IntVar],
    sequence: List[int],
    sequence_in_line: cp_model.IntVar,
):
    line_len = len(line)
    seq_len = len(sequence)

    sequence_matches: List[cp_model.IntVar] = []

    for i in range(line_len - seq_len + 1):
        sequence_match = model.NewBoolVar(f'match_{i}')

        char_matches = []
        for j in range(seq_len):
            char_match = model.NewBoolVar(f'eq_{i}_{j}')
            model.Add(line[i + j] == sequence[j]).OnlyEnforceIf(char_match)
            model.Add(line[i + j] != sequence[j]).OnlyEnforceIf(char_match.Not())
            char_matches.append(char_match)

        model.AddBoolAnd(char_matches).OnlyEnforceIf(sequence_match)
        model.AddBoolOr([c.Not() for c in char_matches]).OnlyEnforceIf(sequence_match.Not())

        sequence_matches.append(sequence_match)

    model.AddBoolOr(sequence_matches).OnlyEnforceIf(sequence_in_line)
    model.AddBoolAnd([s.Not() for s in sequence_matches]).OnlyEnforceIf(sequence_in_line.Not())

def add_sequence_in_grid_constraint(
    model: cp_model.CpModel,
    grid: List[List[cp_model.IntVar]],
    sequence: List[int],
    sequence_in_grid: cp_model.IntVar,
):
    n = len(grid)
    match_rows: List[cp_model.IntVar] = []
    match_cols: List[cp_model.IntVar] = []
    for i in range(n):
        # Check rows
        match_row = model.NewBoolVar(f'match_row_{i}')
        add_sequence_in_line_constraint(model, grid[i, :], sequence, match_row)
        match_rows.append(match_row)

        # Check columns
        match_col = model.NewBoolVar(f'match_col_{i}')
        add_sequence_in_line_constraint(model, grid[:, i], sequence, match_col)
        match_cols.append(match_col)

    model.AddBoolOr(match_rows + match_cols).OnlyEnforceIf(sequence_in_grid)
    model.AddBoolAnd([mr.Not() for mr in match_rows] + [mc.Not() for mc in match_cols]).OnlyEnforceIf(sequence_in_grid.Not())

def constraint_programming(problem: SpongeGridProblem, remaining_time: float = 180):
    start = perf_counter()
    n = problem.n
    char_to_int = {ch: i for i, ch in enumerate(problem.alphabet)}
    int_to_char = {i: ch for i, ch in enumerate(problem.alphabet)}

    model = cp_model.CpModel()

    grid = np.zeros((n, n), dtype=object)
    for i in range(n):
        for j in range(n):
            grid[i, j] = model.NewIntVar(0, len(problem.alphabet)-1, f'int{i}{j}')

    target_matches = []
    for target in problem.targets:
        banned_match = model.NewBoolVar(f'match_target_{"".join(target)}')
        add_sequence_in_grid_constraint(model, grid, [char_to_int[ch] for ch in target], banned_match)
        target_matches.append(banned_match)

    banned_matches = []
    for banned in problem.banned:
        banned_match = model.NewBoolVar(f'match_banned_{"".join(banned)}')
        add_sequence_in_grid_constraint(model, grid, [char_to_int[ch] for ch in banned], banned_match)
        banned_matches.append(banned_match)

    model.Maximize(sum(target_matches) - sum(banned_matches))
    solver = cp_model.CpSolver()
    remaining_time -= perf_counter() - start
    solver.parameters.max_time_in_seconds = remaining_time
    status = solver.Solve(model)

    solution = [[random.choice(problem.alphabet) for _ in range(n)] for _ in range(n)]
    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        # fill grid from solver
        for i in range(n):
            for j in range(n):
                val = solver.Value(grid[i, j])
                solution[i][j] = int_to_char[val]
    else:
        solution = naive_solution(problem)

    return solution


def algo(problem: SpongeGridProblem):
    if problem.n <= 25:
        return constraint_programming(problem)
    return simulated_annealing(problem)