from environment import Environment


class Human:
    def __init__(self):
        pass

    def set_symbol(self, sym):
        self.sym = sym

    def take_action(self, env):
        while True:
            move = input(
                "Enter Coordinates X,Y for your next move (ie: \"0,0\")")
            x, y = move.split(",")
            x = int(x)
            y = int(y)
            if env.is_empty(x, y):
                env.tiles[y, x] = self.sym
                break
            else:
                print("Invalid Move")

    def update(self, env):
        pass

    def update_state_history(self, s):
        pass
