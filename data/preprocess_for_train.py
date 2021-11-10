import sys
import pandas as pd
import re
import copy
import string

regex_1=r"\[\[(?:(?!\[\[|\]\])[\s\S])*\]\]"
regex_2=r"\[\[(.*)\|(.*)\]\]"

def remove_wiki_annotation(text):
    matches = re.findall(regex_1,text)
    for match in matches:
        replace_string = re.sub(regex_2, r"\2", match)
        text = text.replace(match, replace_string)
    return text


dataset_path = sys.argv[1]
df = pd.read_csv(dataset_path)
print("Before removing duplicates {}".format(len(df)))
df = df.drop_duplicates()
print("After removing duplicates {}".format(len(df)))

df = df[df["text_b"] != "none none"]
print("After removing duplicates {}".format(len(df)))

df["text_b"] = df["text_b"].apply(remove_wiki_annotation)

df.to_csv("cleaned_" + dataset_path)
