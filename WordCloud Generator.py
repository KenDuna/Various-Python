import tkinter as tk
from tkinter import filedialog
from os import path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

## If you are using my Reddit Scraper, it saves the .txt files with utf-8 encoding (this prevents errors). 
## You will need to convert the .txt to ansi encoding to use this Word Cloud generator.

root = tk.Tk()
root.withdraw()
file_path = filedialog.askopenfilename()
file = open(file_path, 'rt')
text = file.read()
file.close()

max_words = input('Enter the maximum number of words you wish to have in the Word Cloud:\n')
exclusion = input('Enter any words you wish to exclude (separate by a space, not case sensitive):\n')
exclusion = exclusion.split()
stopwords = set(STOPWORDS)
stopwords.update(exclusion)

wordcloud = WordCloud(max_words = int(max_words), background_color='white', stopwords = stopwords).generate(text)

plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()


