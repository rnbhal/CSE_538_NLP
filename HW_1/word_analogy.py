import os
import pickle
import numpy as np
import scipy.spatial.distance as sp


model_path = './models/'
# loss_model = 'cross_entropy'
loss_model = 'nce'

model_filepath = os.path.join(model_path, 'word2vec_%s_7th.model'%(loss_model))

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


input_file = open('word_analogy_dev.txt', 'r')
# output_file = open('cross_output.txt', 'w')
output_file = open('nce_output.txt', 'w')
result = ""

for line in input_file:
	left_pairs, right_pairs = line.split("||")
	right_tuple = right_pairs.strip().split(",")
	left_tuple = left_pairs.strip().split(",")
	cosine_scores = []
	right_avg = []
	for tup in left_tuple:
			exa, choi = tup.strip().split(":")
			word1 = exa[1:]
			word2 = choi[:-1]
			word_embedding1 = embeddings[dictionary[word1]]
			word_embedding2 = embeddings[dictionary[word2]]
			right_avg.append(np.subtract(word_embedding1, word_embedding2))
	
	right_avg = np.mean(right_avg,axis=0)

	for tup in right_tuple:
		exa, choi = tup.strip().split(":")
		word1 = exa[1:]
		word2 = choi[:-1]
		word_embedding1 = embeddings[dictionary[word1]]
		word_embedding2 = embeddings[dictionary[word2]]
		diff = np.subtract(word_embedding1, word_embedding2)
		cosine_scores.append(1- sp.cosine(right_avg, diff))
	max_idx = cosine_scores.index(max(cosine_scores))
	min_idx = cosine_scores.index(min(cosine_scores))
	#Selected least first and most second as mentioned in the word_analogy_dev_mturk_answers.txt
	result += right_pairs.strip().replace(","," ")+" " + right_tuple[min_idx].strip()+" "+right_tuple[max_idx].strip()+"\n"

output_file.write(result)
output_file.close()