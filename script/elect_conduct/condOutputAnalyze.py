import arpaca.elect_conduct.analyzer as aea
import matplotlib.pyplot as plt

conductivity_list = aea.CondAnalyze()
figure, axis = plt.subplots()
aea.CondPlot(conductivity_list.df, 'error', figure, axis)
aea.CondPlot(conductivity_list.df, 'bubble', figure, axis)
plt.tight_layout()
plt.show()
