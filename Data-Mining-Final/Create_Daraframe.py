import pandas as pd
from IPython.core.display import display

df = pd.read_json('C:/Users/sadik/Downloads/Data-Mining/Youtube-Comment-Sentiment-Analysis-master/Youtube-Comment'
                  '-Sentiment-Analysis-master/comments/theRadBrad_stats.json')

df.to_csv('C:/Users/sadik/Downloads/stats.csv')
display(df)
