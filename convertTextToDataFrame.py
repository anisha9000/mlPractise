import os
import pandas as pd

targetdir = "/home/anisha/temp/tcss555/training/text/"
'''
filelist = os.listdir(targetdir)
#print(filelist)
df_list = [pd.read_table(file) for file in filelist]
print(df_list)
big_df = pd.concat(df_list)
big_df.describe()


file_names = []
data_frames = []
for filename in os.listdir(targetdir):
    name = os.path.splitext(filename)[0]
    file_names.append(name)
    df = pd.read_csv(filename, header=None)
    df.rename(columns={0: name}, inplace=True)
    data_frames.append(df)

combined = pd.concat(data_frames, axis=1)
combined.describe()

df = pd.DataFrame({'text' : ['one', 'two', 'This is very long string very long string very long string veryvery long string']})
print(df)

'''

data = []
path = "/home/anisha/temp/tcss555/training/text/"
files = os.listdir(path)
print(files)
for f in files:
  print("inside for")
  with open (f, "r") as myfile:
    data.append(myfile.read())
    print("data appended from file")

df = pd.DataFrame(data)
print(df)
