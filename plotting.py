import matplotlib.pyplot as plt
import seaborn as sns

def graficar_histograma(data, column, savefig_name=None):
    sns.histplot(data=data, x=column)

    if savefig_name:
        plt.savefig(savefig_name)

    plt.show()
