import pandas as pd
df = pd.read_csv("EnoughTrain.csv")
df["combined_text"] = df['text_a'] + " EVIDENCE " + df['text_b']
print(df.head())
imdb_str = " <|endoftext|> ".join(df['combined_text'].tolist())

with open ('evidences.txt', 'w') as f:
        f.write(imdb_str)
