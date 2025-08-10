from pulp import LpProblem, LpMaximize, LpVariable, LpStatus, lpSum

# Sample player data

players = [(0, 'QB', 50.0, 8000), (1, 'QB', 30.0, 7500),
           (2, 'RB', 25.0, 6000), (3, 'RB', 18.0, 5400), (4, 'RB', 12.0, 5200),
	   	   (5, 'WR', 16.0, 5800), (6, 'WR', 15.0, 5200), (7, 'WR', 13.0, 4900), (8, 'WR', 9.0, 4500),
	       (9, 'TE', 12.0, 4000), (10, 'TE', 10.0, 3800),  (11, 'TE', 10.0, 3200),
	       (12, 'DST', 9.0, 2200), (13, 'DST', 6.0, 1000)]

# Problem parameters
budget = 100000
team_size = 9
positional_reqs = {'QB': 1, 'RB': 2, 'WR': 3, 'TE': 2, 'DST': 1}

# Create PuLP problem
prob = LpProblem("NFL_Player_Selection", LpMaximize)

# Variables
x = {i: LpVariable(f"x_{i}", cat='Binary') for i in range(len(players))}

# Objective: Maximize points
prob += lpSum(players[i][2] * x[i] for i in range(len(players)))

# Salary constraint
prob += lpSum(players[i][3] * x[i] for i in range(len(players))) <= budget

# Positional constraints
for pos, n_k in positional_reqs.items():
	prob += lpSum(x[i] for i in range(len(players)) if players[i][1] == pos) == n_k


# Team size constraint
prob += lpSum(x[i] for i in range(len(players))) == team_size

prob.solve()

# Output results
print("Status:", LpStatus[prob.status])
selected_players = [i for i in range(len(players)) if x[i].value() == 1]
total_points = sum(players[i][2] for i in selected_players)
total_salary = sum(players[i][3] for i in selected_players)
positions_count={}


for i in selected_players:
	pos = players[i][1]
        if pos in positions_count:
		positions_count[pos] += 1
	else:
		positions_count[pos] = 1
	print(f"Player {i} ({pos}): Points={players[i][2]}, Salary={players[i][3]}")

print(f"Total Points: {total_points}")
print(f"Total Salary: {total_salary}")
print(f"Positions: {positions_count}")
