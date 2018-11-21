import numpy as np

def run_viterbi(emission_scores, trans_scores, start_scores, end_scores):
    """Run the Viterbi algorithm.

    N - number of tokens (length of sentence)
    L - number of labels

    As an input, you are given:
    - Emission scores, as an NxL array
    - Transition scores (Yp -> Yc), as an LxL array
    - Start transition scores (S -> Y), as an Lx1 array
    - End transition scores (Y -> E), as an Lx1 array

    You have to return a tuple (s,y), where:
    - s is the score of the best sequence
    - y is the size N array/seq of integers representing the best sequence.
    """
    L = start_scores.shape[0]
    assert end_scores.shape[0] == L
    assert trans_scores.shape[0] == L
    assert trans_scores.shape[1] == L
    assert emission_scores.shape[1] == L
    N = emission_scores.shape[0]


    #viterbi matrix having the score of going to i,j where i is the word & jth tag
    dp_scores = np.zeros((emission_scores.shape[0],emission_scores.shape[1]), dtype=np.float)
    # dp_scores = np.add(dp_scores, emission_scores)

    # dp_scores = np.zeros_like(emission_scores)
    # best_sequence = np.zeros_like(emission_scores)
    # print(start_scores.shape, emission_scores.shape)
    
    #Storing the best sequence 
    best_sequence = np.zeros((emission_scores.shape[0],emission_scores.shape[1]), dtype=np.int32)
    # emission_scores = np.add(emission_scores, start_scores)
    # trans_scores = np.add(trans_scores, start_scores)


    #add start score to dp matrix + emission score
    for i in range(0,L):
        dp_scores[0][i] = start_scores[i] + emission_scores[0][i]

    #ith word/example
    for i in range(1, N):
        # jth tag
        for j in range(L):
            dp_scores[i][j] = -999
            #for each label t
            #transition from all labels in previous column to next column and taking the max of the all scores
            #   1 - 
            #   2 - max(1,2,3) = i,j
            #   3 - 
            for j_each in range(L):
                #transition from j_each to j
                #calculate each score & store the max out of them
                max_scores = dp_scores[i-1][j_each] + trans_scores[j_each][j]
                if max_scores > dp_scores[i][j]:
                    dp_scores[i][j] = max_scores
                    #label chosen for sequence so that we can iterate it backwards to get the sequence
                    best_sequence[i][j] = j_each

            dp_scores[i][j] += emission_scores[i][j]
        

    # print(emission_scores.shape)
    # dp_scores[0] = np.add(emission_scores[0], start_scores.T)
    # print(dp_scores[0].shape, trans_scores.shape)


    #best last index remaining
    viterbi_index = [np.argmax(dp_scores[-1] + end_scores)]
    #best viterbi score
    viterbi_score = np.max(dp_scores[-1] + end_scores)
    # print(viterbi_index, viterbi_score)

    for reverse_begin in reversed(best_sequence[1:]):
        #viterbi_index[-1] is the max last index being appended in last iteration
        #reverse_begin[viterbi_index[-1]] is the current last index
        viterbi_index.append(reverse_begin[viterbi_index[-1]])

    #best sequence in reverse order so reverse it to get in the right order
    viterbi_index.reverse()
    # viterbi_score = np.max(dp_scores[-1] + end_scores)
    # print (viterbi_score, len(viterbi_index))
    return (viterbi_score, viterbi_index)

    # y = []
    # for i in xrange(N):
    #     # stupid sequence
    #     y.append(i % L)
    # # score set to 0
    # return (0.0, y)
