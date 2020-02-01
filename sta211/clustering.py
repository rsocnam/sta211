from minisom import MiniSom
import matplotlib.pyplot as plt


def kohonen(data: list, digits: list, output_directory: str):
    som = MiniSom(10, 10, len(data[0]))
    som.train_random(data, 1000)

    plt.figure(figsize=(8, 8))
    for x, d in zip(data, digits):
        winner = som.winner(x)
        plt. text(winner[0]+.5,  winner[1]+.5,  str(d),
              color=plt.cm.rainbow(d / 10.), fontdict={'weight': 'bold',  'size': 11})

    plt.axis([0, som.get_weights().shape[0], 0,  som.get_weights().shape[1]])
    plt.savefig("{}/som.png".format(output_directory))