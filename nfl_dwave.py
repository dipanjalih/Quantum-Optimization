import dimod
import numpy as np
from dwave.system import DWaveSampler, EmbeddingComposite


# Sample player data




# Problem parameters
budget = 100000
team_size = 9
positional_reqs = {'QB': 1, 'RB': 2, 'WR': 3, 'TE': 2, 'DST': 1}



# Penalty weights
alpha = 1.0  # Objective
beta = 10.0  # Salary
gamma = 100.0  # Positional
delta = 100.0  # Team size


# Initialize QUBO dictionary
N = len(players)
Q = {(i, i): 0.0 for i in range(N)}
Q.update({(i, j): 0.0 for i in range(N) for j in range(i + 1, N)})

#print(Q)

# Points
for i in range(N):
        Q[(i, i)] -= alpha * players[i][2]

# Salary or cost constraint
for i in range(N):
    Q[(i, i)] += beta * players[i][3] ** 2
    Q[(i, i)] -= 2 * beta * budget * players[i][3]
    for j in range(i + 1, N):
        Q[(i, j)] += 2 * beta * players[i][3] * players[j][3]

# Positional constraint
for pos, n_k in positional_reqs.items():
    for i in range(N):
        if players[i][1] == pos:
            Q[(i, i)] += gamma * (1 - 2 * n_k)
            for j in range(i + 1, N):
                if players[j][1] == pos:
                    Q[(i, j)] += 2 * gamma


# Team size constraint
for i in range(N):
    Q[(i, i)] += delta * (1 - 2 * team_size)
    for j in range(i + 1, N):
        Q[(i, j)] += 2 * delta

print(Q)

# Solve using D-Wave
sampler = EmbeddingComposite(DWaveSampler())
bqm = dimod.BinaryQuadraticModel.from_qubo(Q)
response = sampler.sample(bqm, num_reads=1000)

best_sample = response.first.sample
print(best_sample)
selected_players = [i for i in range(N) if best_sample[i] == 1]

# Output results
print("Selected Players:")
total_points = 0
total_salary = 0
positions = {}
for i in selected_players:
    pos = players[i][1]
    positions[pos] = positions.get(pos, 0) + 1
    total_points += players[i][2]
    total_salary += players[i][3]
    print(f"Player {i} ({pos}): Points={players[i][2]}, Salary={players[i][3]}")
print(f"Total Points: {total_points}")
print(f"Total Salary: {total_salary}")
print(f"Positions: {positions}")
