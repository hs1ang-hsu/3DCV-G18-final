import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


emotion_name = ['Neutral', 'Anger', 'Disgust', 'Fear', 'Happiness', 'Sadness', 'Surprise']
def draw_emotion_histogram(data, name):
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.bar(emotion_name, data)

    ax.set(xlabel='Emotions', ylabel='Count')
    ax.set_title('Histogram of emotions in the test dataset')
    ax.grid(True)
    
    plt.savefig(name + '.png')

def draw_emotion_evaluate(data, name):
    corr = pd.DataFrame(data=data, columns=emotion_name, index=emotion_name)
    # mask = np.triu(np.ones_like(corr, dtype=bool))
    
    fig, ax = plt.subplots(figsize=(12, 10))
    cmap = sns.light_palette("red", as_cmap=True)
    ax = sns.heatmap(corr, cmap=cmap, annot=True, fmt='d', #vmax=3,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})
    plt.savefig(name + '.png')


if __name__ == '__main__':
    train = [18721, 18129, 16039, 16035, 15083, 17213, 15215]
    train_eval = [
        [18721, 499, 98, 290, 10, 300, 298],
        [0, 1801, 1702, 1308, 626, 1048, 1559],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 2779, 9558, 1926, 12897, 932, 2601],
        [0, 13050, 4681, 12511, 1550, 14933, 10757],
        [0, 0, 0, 0, 0, 0, 0]
    ]
    
    test = [3150, 2208, 1371, 1163, 2260, 2522, 1439]
    test_eval = [
        [3150, 0, 16, 10, 0, 0, 33],
        [0, 248, 169, 52, 165, 100, 80],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 281, 823, 174, 1735, 51, 142],
        [0, 1679, 363, 927, 360, 2371, 1184],
        [0, 0, 0, 0, 0, 0, 0]
    ]
    
    draw_emotion_histogram(train, 'train_hist')
    draw_emotion_histogram(test, 'test_hist')
    draw_emotion_evaluate(test_eval, 'test_evaluation')