class Environment:
    def __init__(self):
        # Generate board in a matrix
        self.tiles = [[None]*3 for _ in range(3)]
        self.won = False

    def draw(self):
        for row in self.tiles:
            temp = ""
            for tile in row:
                temp += "- " if tile == None else tile + " "
            print(temp)

    def update_and_check_for_winner(self, x, y, char):
        self.tiles[y][x] = char

        # Horizontal win
        if self.tiles[y].count(char) == len(self.tiles[y]):
            self.won = True
            return True

        # Vertical win
        vert = [row[x] for row in self.tiles]
        if vert.count(char) == len(vert):
            self.won = True
            return True

        # Slant win
        winP = True
        winN = True
        for i in xrange(3):
            if winN:
                if self.tiles[2-i][2-i] is not char:
                    winN = False
            if winP:
                if self.tiles[2-i][i] is not char:
                    winP = False

        self.won = winN or winP
        return self.won


env = Environment()
env.update_and_check_for_winner(0, 2, "O")
env.update_and_check_for_winner(1, 1, "O")
env.update_and_check_for_winner(2, 0, "O")
env.draw()
print(env.won)
# env.draw()
