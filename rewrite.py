import random 

class maze:
    def __init__(self, maze, start, end, curr_pos):
        self.start = tuple(start)
        self.maze = maze
        self.end = tuple(end)
        self.curr_position = tuple(curr_pos)
        self.row = len(maze)
        self.col = len(maze[0])

    def in_bounds(self, r, c):
        if 0 <= r < self.row and 0 <= c < self.col:
            if self.maze[r][c] == 0:
                return True
        return False
    
    def reset(self):
        self.curr_position = self.start
        return (self.curr_position)

    def move(self, r, c, action):
        done = False
        actions = {0: (1, 0), 1: (-1, 0), 2: (0, 1), 3: (0, -1)}
        if action not in actions:
            return
        
        dr, dc = actions[action]

        new_pos_r, new_pos_c = (r + dr, c + dc)
        reward = -1

        if self.in_bounds(new_pos_r, new_pos_c):
            self.curr_position = (new_pos_r, new_pos_c)
            if self.curr_position == self.end:
                reward = 10
                done = True
        else:
            self.curr_position = (r, c)
            reward = -10

        return (self.curr_position, reward, done)
    
# -- Parameters for Bellman-Ford 
Q_TABLE = {}
learning_rate = 0.1
epsilon = 1
discount_factor = 0.95
epsilon_decay = 0.99
min_epsilon= 0.001 
iterations = 100

# -- Initialization

start = (0, 0)
end = (3, 5)

make_maze = [[0] * 4 for n in range(6)] #6 x 4, 6 rows, 4 cols



new_maze = maze(make_maze, start, end, start)
ACTIONS = [0, 1, 2, 3]
for r in range(new_maze.row):
    for c in range(new_maze.col):
        Q_TABLE[(r, c)] = {action : 0 for action in ACTIONS} 

# for i in Q_TABLE[(0, 0)]:
    # print(i, Q_TABLE[(0, 0)][i]) 
for i in range(iterations):
    state = new_maze.reset()
    cur_r, cur_c = state
    done = False
    curr_epsilon = max(min_epsilon, epsilon  * epsilon_decay ** i)
    moves = 0

    while not done:
        num = random.random()

        if num <= curr_epsilon: #Explore
            
            action = random.randint(0, 3) #In theory this could be a pmf weighted with how good each of the other options are
        else: #Greedy
            mx = Q_TABLE[state][0]
            action = 0
            for i in Q_TABLE[state]:
                if Q_TABLE[state][i] > mx:
                    action = i
        moves += 1

        new_state, reward, done = new_maze.move(cur_r, cur_c, action)

        old_q_state = Q_TABLE[state][action]
        next_max_q = max(Q_TABLE[new_state].values())
        updated_q_value = old_q_state + learning_rate * (reward + discount_factor * next_max_q - old_q_state)

        Q_TABLE[state][action] = updated_q_value

        state = new_state
        
        if moves >= 20:
            done = True

print("I am done training, and this is my Q_table", Q_TABLE)

                

                

                 

    


    

