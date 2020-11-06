import gensim


model=gensim.models.KeyedVectors.load_word2vec_format("../GoogleNews-vectors-negative300.bin", binary=True)

import re
import codecs

# удаление лишних символов из текста в заданом диапазоне
def preprocess_text(text):
    text = re.sub('[^a-zA-Zа-яА-Я1-9]+', ' ', text)
    text = re.sub(' +', ' ', text) # несколько пробелов заменяются на один
    return text.strip() # несколько пробелов в начале и в конце удаляются

# символы раскодироваются для чтения
# перезапись слов для анализа в кодируемую форму
def prepare_for_w2v(filename_from, filename_to, lang):
    raw_text = codecs.open(filename_from, "r", encoding='windows-1251').read() # прочитать к нормальной форме
    with open(filename_to, 'w', encoding='utf-8') as f: # запись в файл векторов слов
        for sentence in nltk.sent_tokenize(raw_text, lang):    # преобразуем в векторы
            print(preprocess_text(sentence.lower()), file=f)


import multiprocessing
from gensim.models import Word2Vec


def train_word2vec(filename):
    data = gensim.models.word2vec.LineSentence(filename)
    return Word2Vec(data, size=200, window=5, min_count=5, workers=multiprocessing.cpu_count())


keys = ['Paris', 'Python', 'Sunday', 'Tolstoy', 'Twitter', 'bachelor', 'delivery', 'election', 'expensive',
        'experience', 'financial', 'food', 'iOS', 'peace', 'release', 'war']

embedding_clusters = []
word_clusters = []
for word in keys:
    embeddings = []
    words = []
    for similar_word, _ in model.most_similar(word, topn=30):
        words.append(similar_word)
        embeddings.append(model[similar_word])
    embedding_clusters.append(embeddings)
    word_clusters.append(words)

#n_components — количество компонентов, т.е., размерность пространства значений;
#perplexity — перплексия, значение которой в t-SNE можно приравнять к эффективному количеству соседей. Она родственна количеству ближайших соседей, которое используется в других моделях, обучающихся на базе многообразий (см. картинку выше). Ее значение рекомендуется [1] устанавливать в диапазоне 5—50;
#init — тип первоначальной инициализации векторов.
tsne_model_en_2d = TSNE(perplexity=15, n_components=2, init='pca', n_iter=3500, random_state=32)
embedding_clusters = np.array(embedding_clusters)
n, m, k = embedding_clusters.shape
embeddings_en_2d = np.array(tsne_model_en_2d.fit_transform(embedding_clusters.reshape(n * m, k))).reshape(n, m, 2)


# скрипты для построения двумерного графа с помощью matplotlib
from sklearn.manifold import TSNE

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
#% matplotlib inline


def tsne_plot_similar_words(labels, embedding_clusters, word_clusters, a=0.7):
    plt.figure(figsize=(16, 9))
    colors = cm.rainbow(np.linspace(0, 1, len(labels)))
    for label, embeddings, words, color in zip(labels, embedding_clusters, word_clusters, colors):
        x = embeddings[:,0]
        y = embeddings[:,1]
        plt.scatter(x, y, c=color, alpha=a, label=label)
        for i, word in enumerate(words):
            plt.annotate(word, alpha=0.5, xy=(x[i], y[i]), xytext=(5, 2), 
                         textcoords='offset points', ha='right', va='bottom', size=8)
    plt.legend(loc=4)
    plt.grid(True)
    plt.savefig("f/г.png", format='png', dpi=150, bbox_inches='tight')
    plt.show()


tsne_plot_similar_words(keys, embeddings_en_2d, word_clusters)
# Иногда необходимо построить не отдельные кластеры слов, а весь словарь.
prepare_for_w2v(r"C:\Users\Nikita Permikov\source\repos\NETWORK\analysis texts Tolstoy\PythonApplication1\1.txt", r"C:\Users\Nikita Permikov\source\repos\NETWORK\analysis texts Tolstoy\PythonApplication1\train.txt", 'russian')
model_ak = train_word2vec('train.txt')

words = []
embeddings = []
for word in list(model_ak.wv.vocab):
    embeddings.append(model_ak.wv[word])
    words.append(word)
    
tsne_ak_2d = TSNE(n_components=2, init='pca', n_iter=3500, random_state=32)
embeddings_ak_2d = tsne_ak_2d.fit_transform(embeddings)
def tsne_plot_2d(label, embeddings, words=[], a=1):
    plt.figure(figsize=(16, 9))
    colors = cm.rainbow(np.linspace(0, 1, 1))
    x = embeddings[:,0]
    y = embeddings[:,1]
    plt.scatter(x, y, c=colors, alpha=a, label=label)
    for i, word in enumerate(words):
        plt.annotate(word, alpha=0.3, xy=(x[i], y[i]), xytext=(5, 2), 
                     textcoords='offset points', ha='right', va='bottom', size=10)
    plt.legend(loc=4)
    plt.grid(True)
    plt.savefig("grafic.png", format='png', dpi=150, bbox_inches='tight')
    plt.show()


tsne_plot_2d('Преступление и наказание', embeddings_ak_2d, a=0.1)


prepare_for_w2v(r"C:\Users\Nikita Permikov\source\repos\NETWORK\analysis texts Tolstoy\PythonApplication1\1.txt", r"C:\Users\Nikita Permikov\source\repos\NETWORK\analysis texts Tolstoy\PythonApplication1\train.txt", "russian")
model_wp = train_word2vec("train.txt")

words_wp = []
embeddings_wp = []
for word in list(model_wp.wv.vocab):
    embeddings_wp.append(model_wp.wv[word])
    words_wp.append(word)
    
tsne_wp_3d = TSNE(perplexity=30, n_components=3, init='pca', n_iter=3500, random_state=12)
embeddings_wp_3d = tsne_wp_3d.fit_transform(embeddings_wp)
from mpl_toolkits.mplot3d import Axes3D


def tsne_plot_3d(title, label, embeddings, a=1):
    fig = plt.figure()
    ax = Axes3D(fig)
    colors = cm.rainbow(np.linspace(0, 1, 1))
    plt.scatter(embeddings[:, 0], embeddings[:, 1], embeddings[:, 2], c=colors, alpha=a, label=label)
    plt.legend(loc=4)
    plt.title(title)
    plt.show()


tsne_plot_3d('Visualizing Embeddings using t-SNE', 'Преступление и наказание', embeddings_wp_3d, a=0.1)
