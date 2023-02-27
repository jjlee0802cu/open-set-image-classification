import matplotlib.pyplot as plt
import pandas as pd

'''
Creates a few diagrams that I used in my project report
to better explain the approach that I used
'''

bars = pd.DataFrame(dict(data=[0.1, 0.05, 0.08, 0.62, 0.1, 0.05]))
line = pd.DataFrame(dict(data=[0.5, 0.5, 0.5, 0.5, 0.5, 0.5]))

fig, ax = plt.subplots()
bars['data'].plot(kind='bar')
line['data'].plot(kind='line', color='black', linestyle='dashed')

plt.ylim(0,1)
plt.xlabel("Known classes")
plt.ylabel("Softmax confidence")
plt.savefig('./plots/diagram1.png')

plt.clf()
bars = pd.DataFrame(dict(data=[0.2, 0.16, 0.05, 0.42, 0.12, 0.05]))
line = pd.DataFrame(dict(data=[0.5, 0.5, 0.5, 0.5, 0.5, 0.5]))

fig, ax = plt.subplots()
bars['data'].plot(kind='bar')
line['data'].plot(kind='line', color='black', linestyle='dashed')

plt.ylim(0,1)
plt.xlabel("Known classes")
plt.ylabel("Softmax confidence")
plt.savefig('./plots/diagram2.png')

