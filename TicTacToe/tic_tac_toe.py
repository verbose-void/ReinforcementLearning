class Environment:
    def __init__(self, size=3):
        # Generate board in a matrix
        self.tiles = [[None]*size for _ in range(size)]
        self.winner = None
        self.size = size

    def draw(self):
        for row in self.tiles:
            temp = ""
            for tile in row:
                temp += "- " if tile == None else tile + " "
            print(temp)

    def update_and_get_winner(self, x, y, char):
        # If already won, ignore all inputs and return winner.
        if not self.winner == None:
            return self.winner

        self.tiles[y][x] = char

        # Horizontal win
        if self.tiles[y].count(char) == len(self.tiles[y]):
            self.winner = char
            return True

        # Vertical win
        vert = [row[x] for row in self.tiles]
        if vert.count(char) == len(vert):
            self.winner = char
            return True

        # Slant win
        winP = True
        winN = True
        for i in xrange(self.size):
            if winN:
                if self.tiles[self.size-i-1][self.size-i-1] is not char:
                    winN = False
            if winP:
                if self.tiles[self.size-i-1][i] is not char:
                    winP = False

        self.winner = char if winN or winP else None
        return self.winner


env = Environment()
env.update_and_get_winner(0, 0, "X")
env.update_and_get_winner(1, 1, "X")
env.update_and_get_winner(2, 2, "X")
env.draw()
print(env.winner)
# env.draw()
