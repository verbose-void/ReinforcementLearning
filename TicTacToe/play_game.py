import environment as enviro
from human import Human
from agent import Agent
import numpy as np


def play_game(p1, p2, env, draw=False):
    p1_turn = True
    while not env.game_over():
        if draw:
            # Draw the board for the player who wants to see it
            if draw == 1 and p1_turn:
                env.draw()
            elif draw == 2 and not p1_turn:
                env.draw()

        if p1_turn:
            p1.take_action(env)
        else:
            p2.take_action(env)

        state = env.get_state()
        p1.update_state_history(state)
        p2.update_state_history(state)

        # Swap Turns
        p1_turn = not p1_turn

    if draw:
        if env.winner == None:
            print("Draw!")
        elif env.winner == 1:
            print("The winner is O!")
        elif env.winner == -1:
            print("The winner is X!")
        env.draw()

    p1.update(env)
    p2.update(env)


if __name__ == "__main__":
    # train the agents
    p1 = Agent(eps=0.05)
    p2 = Agent(eps=0.05)

    # set initial V for p1 and p2
    env = enviro.Environment()
    state_winner_triples = enviro.get_state_hash_and_winner(env)

    ans = input("Load training data from file? (y/n): ")
    if ans == "y":
        try:
            Vx = np.load("p1_value_data.npy")
            print("Training data imported from \"p1_value_data.npy\"")
        except IOError:
            Vx = enviro.initialV_x(env, state_winner_triples)
            print("Training data file \"p1_value_data.npy\" was not found.")
            print("Generating initial values for P1.")

        try:
            Vo = np.load("p2_value_data.npy")
            print("Training data imported from \"p2_value_data.npy\"")
        except IOError:
            Vo = enviro.initialV_o(env, state_winner_triples)
            print("Training data file \"p2_value_data.npy\" was not found.")
            print("Generating initial values for P2.")
    else:
        Vo = enviro.initialV_o(env, state_winner_triples)
        Vx = enviro.initialV_x(env, state_winner_triples)

    p1.setV(Vx)
    p1.set_symbol(env.x)

    p2.setV(Vo)
    p2.set_symbol(env.o)

    new_ans = "y"
    if ans == "y":
        new_ans = input("Would you like to continue training? (y/n): ")

    if new_ans == "y":
        T = 10000
        for t in range(T):
            if t % 200 == 0:
                print(t)
            play_game(p1, p2, env.reset())

    human = Human()
    human.set_symbol(p2.sym)
    p1.set_verbose(True)

    while True:
        res = input("Would you like to play? (y/n): ")
        if res == "y":
            play_game(p1, human, env.reset(), draw=2)
        else:
            break

    ans = input("Would you like to export the training data? (y/n): ")
    if ans == "y":
        np.save("p1_value_data.npy", p1.V)
        np.save("p2_value_data.npy", p2.V)
