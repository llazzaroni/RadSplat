import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def main():

    data = ["data/poster/sfm-nerfacto.json", "data/poster/radsplat-nerfacto-3k-it.json","data/poster/result-500k-sobel-poster.json"]
    names = ["sfm", "radsplat-nerfact-3k", "sobel-500k"]
    tables = []

    for i in data:
        with open(i, "r") as f:
            table = pd.read_json(f)
            table['iter'] = table.index.values
            table['iter'] = table['iter'].map(lambda x : (x+1)*100)
            tables.append(table)
            print(table)


    print(tables[0])

    fig, axs = plt.subplots(2,2)

    for ax, metric in zip(axs.flat, tables[0].columns[:-1]):
        for name, table in zip(names, tables):
            ax.plot(table['iter'], table[metric], marker = 'o', label = name)
            ax.set_title(metric)
            ax.legend()

    plt.show()

if __name__ == "__main__":
    main()
