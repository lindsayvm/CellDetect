import pandas as pd

df = {'Answer': [0, 1, 2, 3], 'Question': ["tumor","tumor","lymph","lymph"]}
df = pd.DataFrame(data=df, index=[0, 1, 2, 3])

print(df.loc[df['Question'] =="tumor"])
#for _, p in self.scores[self.scores['Question'] == boxes_question].iterrows()]