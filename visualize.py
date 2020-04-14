from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import matplotlib.pyplot as plt
import pandas as pd

"""
Loading Datasets
"""
df = pd.read_csv("amazon-fine-food-reviews/Reviews.csv")


# Start with one review:
text = df.Text

x = ""
for t in text:
    x = x + t

# Create and generate a word cloud image:
wordcloud = WordCloud(max_font_size=50, max_words=1000,
                      width=1350, height=760).generate(x)
wordcloud.to_file("first_review.png")

# Display the generated image:
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()
