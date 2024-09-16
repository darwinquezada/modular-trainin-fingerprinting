# importing package
import matplotlib
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import numpy as np
import os

# plt.style.use('seaborn-v0_8-paper')
with plt.style.context(("seaborn-pastel",)):

    plt.rcParams["figure.figsize"] = [5, 3]
    plt.rcParams["figure.autolayout"] = True
    
    # plt.rcParams['axes.grid'] = True
    # plt.rcParams['grid.alpha'] = 1
    # plt.rcParams['grid.color'] = "#cccccc"


    # create data
    x = np.arange(3)
    labels = ['TUT3', 'UJI1', 'UTSI']
    cnnloc = [89.90, 94.96, 91.50]
    modular_nn = [88.61, 92.98, 85.83]
    width = 0.45  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width/2, cnnloc, width, label='CNNLoc', color='#92c6ff', edgecolor='black')
    rects2 = ax.bar(x + width/2, modular_nn, width, label='Modular NN', color='#ff9e9a', edgecolor='black')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_xlabel("Datasets", fontsize=11)
    ax.set_ylabel("Floor Hit Rate [%]", fontsize=11)
    ax.set_title("Floor Hit Rate", fontsize=12)
    ax.grid()
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(loc='lower right')
    ax.margins(y=0.1)

    # Set tick font size
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontsize(11)

    def autolabel(rects):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 0),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')


    autolabel(rects1)
    autolabel(rects2)

    fig.tight_layout()

    main_path = os.path.join('report', 'PLOTS')

    if not os.path.exists(main_path):
        os.makedirs(main_path)
        
    plt.savefig(os.path.join(main_path,'floor_hit_rate.pdf'))

    plt.show()