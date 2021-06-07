import numpy as np
import matplotlib
import matplotlib.pyplot as plt


class plots():
    """plotting class"""
    def __init__(self, technics, technics_performance, balancing):
        self.objects = technics
        self.performance = technics_performance
        self.balancing_technic = balancing
        y_pos = np.arange(len(self.objects))
        bars = plt.bar(y_pos, self.performance, align='center', alpha=0.5)
        plt.xticks(y_pos, self.objects)
        plt.ylabel('Procent dokładności')
        plt.title(self.balancing_technic)
        #plt.figure(figsize=(20, 10), dpi=100)
        for bar in bars:
            yval = bar.get_height()
            string = "{:.7f}".format(bar.get_height())
            plt.text(bar.get_x(), yval + .005, string)

        self.plot = plt
    
    def show(self):
        self.plot.show()

    def save(self, name):
        self.plot.savefig(name)
        self.plot.clf()
        

    def fig(self):
        fig = self.plot.gcf()
        fig.set_size_inches(10, 5)
        return fig


