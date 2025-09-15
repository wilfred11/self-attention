import matplotlib.pyplot as plt

def visualise():
    # Create word embeddings
    xs = [0.5, 1.5, 2.5, 6.0, 7.5, 8.0]
    ys = [3.0, 1.2, 0.5, 8.0, 7.5, 5.5]
    words = ['money', 'deposit', 'withdraw', 'nature', 'river', 'water']
    bank = [[4.5, 4.5], [6.7, 6.5]]

    # Create figure
    fig, ax = plt.subplots(ncols=2, figsize=(8,4))

    # Add titles
    ax[0].set_title('Learned Embedding for "bank"\nwithout context')
    ax[1].set_title('Contextual Embedding for\n"bank" after self-attention')

    # Add trace on plot 2 to show the movement of "bank"
    ax[1].scatter(bank[0][0], bank[0][1], c='blue', s=50, alpha=0.3)
    ax[1].plot([bank[0][0]+0.1, bank[1][0]],
               [bank[0][1]+0.1, bank[1][1]],
               linestyle='dashed',
               zorder=-1)

    for i in range(2):
        ax[i].set_xlim(0,10)
        ax[i].set_ylim(0,10)

        # Plot word embeddings
        for (x, y, word) in list(zip(xs, ys, words)):
            ax[i].scatter(x, y, c='red', s=50)
            ax[i].text(x+0.5, y, word)

        # Plot "bank" vector
        x = bank[i][0]
        y = bank[i][1]

        color = 'blue' if i == 0 else 'purple'

        ax[i].text(x+0.5, y, 'bank')
        ax[i].scatter(x, y, c=color, s=50)
    plt.savefig("plots/visual_attention.png")