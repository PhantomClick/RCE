import gensim
import numpy as np
from sklearn.preprocessing import MinMaxScaler

calculate_similarity = False

if calculate_similarity == True:

    model = gensim.models.KeyedVectors.load_word2vec_format('D:/GoogleNews-vectors-negative300.bin', binary=True)


    categories = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']

    similarities = np.zeros(shape=(len(categories),len(categories)),dtype=np.float32)

    # measure similarity between two words: model.similarity('france', 'spain')


    for w_1 in range(0,len(categories)):
        for w_2 in range(0,len(categories)):
            score_1 = model.similarity(categories[w_1], categories[w_2])
            score_2 = model.similarity(categories[w_1],categories[w_2].title())
            score_3 = model.similarity(categories[w_1].title(),categories[w_2])
            score_4 = model.similarity(categories[w_1].title(),categories[w_2].title())

            average_score = (score_1 + score_2 + score_3 + score_4) / 4

            similarities[w_1][w_2] = average_score

    print("Similarity Scores are:", similarities)

    np.save("./saved_models/similarity_matrix.npy",similarities)

else:
    matrix = np.load("./saved_models/similarity_matrix.npy")

    scaler = MinMaxScaler()

    matrix = scaler.fit_transform(matrix)
    matrix = np.around(matrix, 2)

    for i in range(10):
        print(matrix[i,:])