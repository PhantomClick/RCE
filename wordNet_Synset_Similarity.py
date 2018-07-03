from nltk.corpus import wordnet
import itertools as IT
import numpy as np

def calculate_synset_similarity(word1, word2):

    wordFromList1 = wordnet.synsets(word1)[0]
    wordFromList2 = wordnet.synsets(word2)[0]

    s = wordFromList1.path_similarity(wordFromList2)

    return s

calculate_similarity = False

if calculate_similarity == True:

    categories = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']

    similarities = np.zeros(shape=(len(categories),len(categories)),dtype=np.float32)

    # measure similarity between two words: model.similarity('france', 'spain')


    for w_1 in range(0,len(categories)):
        for w_2 in range(0,len(categories)):
            score_1 = calculate_synset_similarity(categories[w_1], categories[w_2])

            similarities[w_1][w_2] = score_1

    print("Similarity Scores are:", similarities)

    np.save("./saved_models/synset_similarity_matrix.npy",similarities)

else:

    matrix = np.load("./saved_models/synset_similarity_matrix.npy")

    matrix = np.around(matrix, 2)

    for i in range(10):
        print(matrix[i,:])