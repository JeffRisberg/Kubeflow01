import os

import pandas as pd
import torch
import json

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

from datasets import Dataset
from transformers import AutoTokenizer, AutoModel
import ast

# return a dataframe that we can use later
def load_articles():
  df = pd.read_csv('../data/kbs_with_entities.csv')
  df = df.reset_index()  # make sure indexes pair with number of rows
  for index, row in df.iterrows():
    list_of_entities = ast.literal_eval(row['entities'])
    key = " ".join([row['title']] + [row['subject']] + list_of_entities)
    df.at[index, 'key'] = key
  return df


df = load_articles()
knowledge_articles_data = Dataset.from_pandas(df)
print(knowledge_articles_data)

preTrained = "sentence-transformers/multi-qa-mpnet-base-dot-v1"
tokenizer = AutoTokenizer.from_pretrained(preTrained)
model = AutoModel.from_pretrained(preTrained)

device = torch.device("cpu")
model.to(device)


# get embedding of the first token, the output is a Torch tensor
def get_embeddings(text_list):
  encoded_input = tokenizer(text_list, padding=True, truncation=True,
                            return_tensors="pt")
  encoded_input = {k: v.to(device) for k, v in encoded_input.items()}
  output = model(**encoded_input)
  clsEmbedding = output.last_hidden_state[:, 0]
  return clsEmbedding


knowledge_articles_data_set = knowledge_articles_data.map(lambda article: {
  "embeddings": get_embeddings(article['key']).detach().cpu().numpy()[0]})
knowledge_articles_data_set.add_faiss_index(column="embeddings")


def get_results(knowledge_articles_data_set, input):
  phrase = " ".join([input['phrase']] + input['entities'])
  embeddingQ = get_embeddings([phrase]).cpu().detach().numpy()
  scores, samples = knowledge_articles_data_set.get_nearest_examples(
       "embeddings", embeddingQ, k=3)
  for i in reversed(range(3)):
     print(samples['content_id'][i], samples['title'][i], scores[i])

inputs = [{'phrase': "transfer shares",
           'entities': ['transfer', 'shares']},
          {'phrase': "Cancel an Option Grant?",
           'entities': ['option', 'grant', 'cancel']},
          {'phrase': "Convertible Notes and SAFE Terms and Definitions",
           'entities': ['convertable', 'note', 'safe', 'terms']},
          {'phrase': "Can I see what was submitted in the 409A request form?",
           'entities': ['409A', 'request']},
          {'phrase': "federal exemption question",
           'entities': ['federal exemptions']},
          {'phrase': "terminate option holders",
           'entities': ['option', 'holders']},
          {'phrase': "can i contact someone to discuss my 409a valuation?",
           'entities': ['409a', 'valuation']},
          {'phrase': "when does Carta send investor requests to the designated CEO",
           'entities': ['investor', 'requests']},
          {'phrase': "how do I approve an Option Exercise",
           'entities': ['Option', 'exercise', 'approve']},
          {'phrase': "Email preferences for Investment Firms",
           'entities': ['email', 'preferences', 'firms']},
          {'phrase': "error accepting grant",
           'entities': ['grant', 'accept', 'error']}
          ]

for input in inputs:
  print("Input phrase:", input['phrase'])
  get_results(knowledge_articles_data_set, input)
  print()
