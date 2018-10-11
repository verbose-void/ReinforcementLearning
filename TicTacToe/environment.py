import numpy as np


class Environment:
    def __init__(self, size=3):
        # Generate board in a matrix
        self.x = -1
        self.o = 1
        self.size = size
        self.num_states = 3**(size*size)
        self.reset()

    def reset(self):
        self.tiles = np.zeros((self.size, self.size))
        self.winner = None
        self.ended = False
        return self

    def reward(self, sym):
        if not self.game_over():
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

    def get_state(self):
        # creates a hash for the current game state
        h = 0
        k = 0
        for y in xrange(self.size):
            for x in xrange(self.size):
                sym = self.tiles[y, x]
                if sym == self.x:
                    v = 1
                elif sym == self.o:
                    v = 2
                else:
                    v = 0
                h += (3**k) * v
                k += 1
        return h


def get_state_hash_and_winner(env, i=0, j=0):
    results = []

    for v in (0, env.x, env.o):
        env.tiles[i, j] = v

        if j == env.size - 1:
            if i == env.size - 1:
                # board is full
                state = env.get_state()
                game_over = env.game_over(force_recalculate=True)
                winner = env.winner
                results.append((state, winner, game_over))
            else:
                results += get_state_hash_and_winner(env, i + 1, 0)
        else:
            results += get_state_hash_and_winner(env, i, j + 1)

    return results


def initialV_x(env, state_winner_triples):
    V = np.zeros(env.num_states)

    for state, winner, game_over in state_winner_triples:
        if game_over:
            if winner == env.x:
                v = 1
            else:
                v = 0
        else:
            v = 0.5
        V[state] = v

    return V


def initialV_o(env, state_winner_triples):
    V = np.zeros(env.num_states)

    for state, winner, game_over in state_winner_triples:
        if game_over:
            if winner == env.o:
                v = 1
            else:
                v = 0
        else:
            v = 0.5
        V[state] = v

    return V


if __name__ == "__main__":
    env = Environment()
    triples = get_state_hash_and_winner(env)
    print(initialV_o(env, triples))
