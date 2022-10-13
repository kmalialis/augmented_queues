import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def create_data(df, memory_size_):
    name_init = 'TwoPatterns_init_mem' + str(memory_size_) + '.csv'
    classes_ = list(set(df['Class'].values))
    df_init = pd.DataFrame()
    for c in classes_:
        dfTemp = df[df['Class'] == c]
        dfTemp = dfTemp[:memory_size_]
        df_init = pd.concat([dfTemp, df_init])
        df_init = df_init.reset_index(drop=True)
    for i_ in range(len(classes_)):
        df_init['Class'] = df_init['Class'].mask(df_init['Class'] == classes_[i_], i_)
    df_init.to_csv(name_init, index=False, header=False)


if __name__ == '__main__':
    dfTrain = pd.read_csv('TwoPatterns_Train.csv')
    dfTest = pd.read_csv('TwoPatterns_TEST.csv')
    random_state = np.random.RandomState(0)
    memory_size = 10

    create_data(dfTrain, memory_size)

    classes = list(set(dfTest['Class'].values))
    for i in range(len(classes)):
        dfTest['Class'] = dfTest['Class'].mask(dfTest['Class'] == classes[i], i)
    name_arr = 'TwoPatterns_arr_mem' + str(memory_size) + '.csv'
    dfTest.to_csv(name_arr, index=False, header=False)

