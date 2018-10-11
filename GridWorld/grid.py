class Grid:
    def __init__(self, height, width, start):
        self.height = height
        self.width = width
        # Starting position on the grid
        self.start = start
        self.i = start[0]
        self.j = start[1]

    def set(self, rewards, actions):
        self.actions = actions
        self.rewards = rewards

    def set_state(self, s):
        self.i = s[0]
        self.j = s[1]

    def current_state(self):
        return (self.i, self.j)

    def is_terminal(self, s):
        return s not in self.actions

    def move(self, action):
        if action in self.actions[(self.i, self.j)]:
            if action == "U":
                self.i -= 1
            elif action == "D":
                self.i += 1
            elif action == "L":
                self.j -= 1
            elif action == "R":
                self.j += 1

        return self.rewards.get((self.i, self.j), 0)

    def draw(self):
        for i in xrange(self.height):
            row = ""
            for j in xrange(self.width):
                if self.start[0] == i and self.start[1] == j:
                    row += " S "
                elif not self.is_terminal((i, j)):
                    row += " - "
                elif (i, j) in self.rewards:
                    row += " " + str(self.rewards.get((i, j))) + " "
                else:
                    row += " X "
            print(row)

    def game_over(self):
        return self.is_terminal(self.current_state())

    def undo_move(self, action):
        if action == "U":
            self.i += 1
        elif action == "D":
            self.i -= 1
        elif action == "L":
            self.j += 1
        elif action == "R":
            self.j -= 1

        assert(self.current_state() in self.all_states())

    def all_states(self):
        return set(self.actions.keys()) | set(self.rewards.keys())


def standard_grid():
    g = Grid(3, 4, (2, 0))
    rewards = {(0, 3): 1, (1, 3): -1}
    actions = {
        (0, 0): ('D', 'R'),
        (0, 1): ('L', 'R'),
        (0, 2): ('L', 'D', 'R'),
        (1, 0): ('U', 'D'),
        (1, 2): ('U', 'D', 'R'),
        (2, 0): ('U', 'R'),
        (2, 1): ('L', 'R'),
        (2, 2): ('L', 'R', 'U'),
        (2, 3): ('L', 'U'),
    }
    g.set(rewards, actions)
    return g


def negative_grid(step_cost=-0.1):
    g = standard_grid()
    g.rewards.update({
        (0, 0): step_cost,
        (0, 1): step_cost,
        (0, 2): step_cost,
        (1, 0): step_cost,
        (1, 2): step_cost,
        (2, 0): step_cost,
        (2, 1): step_cost,
        (2, 2): step_cost,
        (2, 3): step_cost,
    })
    return g


if __name__ == "__main__":
    grid = standard_grid()
    grid.draw()
