import os
import pickle
import numpy as np
import scipy.spatial.distance as sp

model_path = './models/'
loss_model = 'cross_entropy'
# loss_model = 'nce'
output_filename = "nce_similar_words.txt"

model_filepath = os.path.join(model_path, 'word2vec_%s.model'%(loss_model))

dictionary, steps, embeddings = pickle.load(open(model_filepath, 'r'))

"""
==========================================================================

Write code to evaluate a relation between pairs of words.
You can access your trained model via dictionary and embeddings.
dictionary[word] will give you word_id
and embeddings[word_id] will return the embedding for that word.

word_id = dictionary[word]
v1 = embeddings[word_id]

or simply

v1 = embeddings[dictionary[word_id]]

==========================================================================
"""


# input_file = open('word_analogy_dev.txt', 'r')
# output_file = open('cross_output.txt', 'w')
output_file = open(output_filename, 'w')
result = ""

f_embeddings = embeddings[dictionary['first']]
a_embeddings = embeddings[dictionary['american']]
w_embeddings = embeddings[dictionary['would']]

f_score = []
a_score = []
w_score = []

for words in dictionary:
    # print words
    word_embedd = embeddings[dictionary[words]]
    f_score.append( tuple ( ((1- sp.cosine(f_embeddings, word_embedd)) , words) ) )
    a_score.append( tuple ( ((1- sp.cosine(a_embeddings, word_embedd)) , words) ) )
    w_score.append( tuple ( ((1- sp.cosine(w_embeddings, word_embedd)) , words) ) )

f_score = sorted(f_score, key=lambda x: x[0])
a_score = sorted(a_score, key=lambda x: x[0])
w_score = sorted(w_score, key=lambda x: x[0])

max_f_score = f_score[-21:-1]
max_a_score = a_score[-21:-1]
max_w_score = w_score[-21:-1]

result += "first \n"
for tup in max_f_score:
    result += tup[1] + " "
result += "\n"

result += "\namerican \n"
for tup in max_a_score:
    result += tup[1] + " "
result += "\n"

result += "\nwould \n"
for tup in max_w_score:
    result += tup[1] + " "
result += "\n"

output_file.write(result)
output_file.close()

print("Saved outfile: {}".format(output_filename))