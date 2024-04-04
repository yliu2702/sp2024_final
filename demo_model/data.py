from base.helper import *
from base.constant import *
import pandas as pd
from sklearn.model_selection import train_test_split

"""### DATASET
X: statement_clean; justification_clean
Y: label_num; label_tf
"""

data = pd.read_csv(DATA_BASE_DIR + "/LIAR_text_label.csv",index_col = 0)
data = data.sample(frac=1).reset_index(drop=True)
test_size = 0.2
train_df, temp_df = train_test_split(data, test_size=test_size)
val_df, test_df = train_test_split(temp_df, test_size=0.5)
print(train_df.shape, val_df.shape, test_df.shape)

train_df.to_csv(DATA_BASE_DIR + "/model/train.csv",index = False)
val_df.to_csv(DATA_BASE_DIR + "/model/val.csv",index = False)
test_df.to_csv(DATA_BASE_DIR + "/model/test.csv",index = False)


