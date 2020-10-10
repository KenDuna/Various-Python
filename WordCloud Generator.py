import tkinter as tk
from tkinter import filedialog
from os import path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator


root = tk.Tk()
root.withdraw()

file_path = filedialog.askopenfilename()

file = open(file_path, 'rt')
text = file.read()
file.close()

wordcloud = WordCloud(max_words = 50, background_color='white').generate(text)

plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()


