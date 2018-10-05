import numpy as np


class Environment:
    def __init__(self, size=3):
        # Generate board in a matrix
        self.tiles = np.zeros((size, size))
        self.x = -1
        self.o = 1
        self.winner = None
        self.ended = False
        self.size = size
        self.num_states = 3**(size*size)

    def reward(self, sym):
        if game_over():
            return 0
        return 1 if self.winner == sym else 0

    def is_empty(self, x, y):
        return self.tiles[y, x] == 0

    def draw(self):
        for y in xrange(self.size):
            for x in xrange(self.size):
                print " ",
                if self.tiles[y, x] == self.x:
                    print "X",
                elif self.tiles[y, x] == self.o:
                    print "O",
                else:
                    print "-",
            print ""

    def game_over(self, force_recalculate=False):
        if not force_recalculate and self.ended:
            return self.ended

        # Check for Horizontal Win
        for y in xrange(self.size):
            for player in (self.x, self.o):
                if self.tiles[y].sum() == player*self.size:
                    self.ended = True
                    self.winner = player
                    return True

        # Check for Vertical Win
        for x in xrange(self.size):
            for player in (self.x, self.o):
                if self.tiles[:, x].sum() == player*self.size:
                    self.ended = True
                    self.winner = player
                    return True

        # Check for Diagonal Win
        for player in (self.x, self.o):
            if self.tiles.trace() == player*self.size:
                self.ended = True
                self.winner = player
                return True

            if np.fliplr(self.tiles).trace() == player*self.size:
                self.ended = True
                self.winner = player
                return True

        # Check for draw
        self.winner = None

        if np.all((self.tiles == 0) == False):
            self.ended = True
            return True

        return False

        # ... TODO

    def get_state(self):
        # creates a hash for the current game state
        h = 0
        k = 0
        for y in xrange(self.size):
            for x in xrange(self.size):
                sym = self.tiles[y, x]
                if sym is None:
                    v = 0
                elif sym is self.o:
                    v = 1
                elif sym is self.x:
                    v = 2
            h += (3**k) * v
            k += 1
        return h


def get_state_hash_and_winner(env, x=0, y=0):
    results = []

    for v in (None, env.x, env.o):
        env.tiles[y, x] = v

        if y == env.size - 1:
            if x == env.size - 1:
                # board is full
                state = env.get_state()
                game_over = env.game_over(force_recalculate=True)
                winner = env.winner
                results.append((state, winner, game_over))
            else:
                results += get_state_hash_and_winner(env, x + 1, y)
        else:
            results += get_state_hash_and_winner(env, x, y + 1)

    return results


def initialV(env, state_winner_triples, is_for_x):
    V = np.zeros(env.num_states)

    for state, winner in state_winner_triples:
        if winner == env.x:
            v = 1 if is_for_x else 0
        elif winner == env.o:
            v = 0 if is_for_x else 1
        else:
            v = 0.5
        V[state] = v

    return V


if __name__ == "__main__":
    env = Environment()
    env.tiles[0, 0] = env.x
    env.tiles[0, 1] = env.x
    env.tiles[0, 2] = env.o
    env.draw()
    print("Game Over?", env.game_over())
