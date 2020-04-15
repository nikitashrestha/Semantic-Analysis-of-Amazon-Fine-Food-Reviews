import numpy as np
import pandas as pd
from os import path
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import matplotlib.pyplot as plt

# from google.colab import files
# uploaded = files.upload()
from google.colab import drive
drive.mount('/content/gdrive')

"""
Loading Datasets
"""
df = pd.read_csv("gdrive/My Drive/Reviews.csv")


# Start with one review:
text = df.Text

x = ""
for t in text:
    x = x + t

# Create and generate a word cloud image:
wordcloud = WordCloud(max_font_size=50, max_words=1000,
                      width=1350, height=760).generate(x)

# Display the generated image:
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()
wordcloud.to_file("/content/gdrive/My Drive/wordcloud.png")
