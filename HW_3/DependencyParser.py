import collections
import os
import tensorflow as tf
import numpy as np
import pickle
import math
from progressbar import ProgressBar

from DependencyTree import DependencyTree
from ParsingSystem import ParsingSystem
from Configuration import Configuration
import Config
import Util

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
"""
This script defines a transition-based dependency parser which makes
use of a classifier powered by a neural network. The neural network
accepts distributed representation inputs: dense, continuous
representations of words, their part of speech tags, and the labels
which connect words in a partial dependency parse.

This is an implementation of the method described in

Danqi Chen and Christopher Manning. A Fast and Accurate Dependency Parser Using Neural Networks. In EMNLP 2014.

Author: Danqi Chen, Jon Gauthier
Modified by: Heeyoung Kwon (2017)
Modified by: Jun S. Kang (2018 Mar)
"""


class DependencyParserModel(object):

    def __init__(self, graph, embedding_array, Config):

        self.build_graph(graph, embedding_array, Config)

    def build_graph(self, graph, embedding_array, Config):
        """

        :param graph:
        :param embedding_array:
        :param Config:
        :return:
        """

        with graph.as_default():
            self.embeddings = tf.Variable(embedding_array, dtype=tf.float32)

            """
            ===================================================================

            Define the computational graph with necessary variables.
            
            1) You may need placeholders of:
                - Many parameters are defined at Config: batch_size, n_Tokens, etc
                - # of transitions can be get by calling parsing_system.numTransitions()
                
            self.train_inputs = 
            self.train_labels = 
            self.test_inputs =
            ...
            
                
            2) Call forward_pass and get predictions
            
            ...
            self.prediction = self.forward_pass(embed, weights_input, biases_input, weights_output)


            3) Implement the loss function described in the paper
             - lambda is defined at Config.lam
            
            ...
            self.loss =
            
            ===================================================================
            """
            numTrans = parsing_system.numTransitions()
            self.train_inputs = tf.placeholder(tf.int32, shape= (Config.batch_size, Config.n_Tokens))
            self.train_labels = tf.placeholder(tf.int32, shape= (Config.batch_size, numTrans))
            self.test_inputs = tf.placeholder(tf.int32, shape= (Config.n_Tokens))


            #Input for Forward Pass
            train_embed = tf.nn.embedding_lookup(self.embeddings, self.train_inputs)
            train_embed = tf.reshape(train_embed, [Config.batch_size, Config.n_Tokens*Config.embedding_size])
    
            #First Layer weight input
            weights_input = tf.Variable( tf.random_normal([Config.n_Tokens*Config.embedding_size, Config.hidden_size], mean=0, stddev=0.1, dtype=tf.float32) )

            #Second Layer weight input
            # weights_sec_input = tf.Variable( tf.random_normal([Config.hidden_size, Config.hidden_size], mean=0, stddev=0.1, dtype=tf.float32) )

            #Second Layer weight input
            # weights_third_input = tf.Variable( tf.random_normal([Config.hidden_size, Config.hidden_size], mean=0, stddev=0.1, dtype=tf.float32) )



            #First Layer bias input
            biases_input = tf.Variable( tf.random_normal([Config.hidden_size], mean=0, stddev=0.1, dtype=tf.float32) )

            #Second Layer bias input
            # biases_sec_input = tf.Variable( tf.random_normal([Config.hidden_size], mean=0, stddev=0.1, dtype=tf.float32) )

            #Second Layer bias input
            # biases_third_input = tf.Variable( tf.random_normal([Config.hidden_size], mean=0, stddev=0.1, dtype=tf.float32) )




            weights_output = tf.Variable( tf.random_normal([Config.hidden_size, numTrans], mean=0, stddev=0.1, dtype=tf.float32) )




            '''
            Forward Pass for Single Layer
            '''
            self.prediction = self.forward_pass(train_embed, weights_input, biases_input, weights_output)
            

            '''
            Forward Pass for Two Layers
            '''
            # self.prediction = self.forward_pass_sec(train_embed, weights_input, biases_input, weights_sec_input, biases_sec_input, weights_output)
            

            '''
            Forward Pass for Three Layers
            '''
            # self.prediction = self.forward_pass_third(train_embed, weights_input, biases_input, weights_sec_input, biases_sec_input, weights_third_input, biases_third_input, weights_output)
        


            #Get the Softmax Cross Entropy loss for the predicted labels
            soft = tf.nn.softmax_cross_entropy_with_logits_v2(labels = tf.nn.relu(self.train_labels), logits = self.prediction)
            self.loss = tf.reduce_mean(soft)


            #Compute the l2 loss for all the inputs
            l2 = tf.nn.l2_loss(weights_input) + tf.nn.l2_loss(biases_input) + tf.nn.l2_loss(weights_output) + tf.nn.l2_loss(train_embed)
            # l2 = tf.nn.l2_loss(weights_input) + tf.nn.l2_loss(biases_input) + tf.nn.l2_loss(weights_sec_input) + tf.nn.l2_loss(biases_sec_input) + tf.nn.l2_loss(weights_output) + tf.nn.l2_loss(train_embed)
            # l2 = tf.nn.l2_loss(weights_input) + tf.nn.l2_loss(biases_input) + tf.nn.l2_loss(weights_sec_input) + tf.nn.l2_loss(biases_sec_input) + tf.nn.l2_loss(weights_third_input) + tf.nn.l2_loss(biases_third_input) + tf.nn.l2_loss(weights_output) + tf.nn.l2_loss(train_embed)
            
            l2 = tf.math.scalar_mul(Config.lam , l2)


            #Compute the total loss = Softmax + l2
            self.loss = self.loss + l2

            optimizer = tf.train.GradientDescentOptimizer(Config.learning_rate)
            grads = optimizer.compute_gradients(self.loss)
            clipped_grads = [(tf.clip_by_norm(grad, 5), var) for grad, var in grads]
            self.app = optimizer.apply_gradients(clipped_grads)

            # For test data, we only need to get its prediction
            test_embed = tf.nn.embedding_lookup(self.embeddings, self.test_inputs)
            test_embed = tf.reshape(test_embed, [1, -1])

            self.test_pred = self.forward_pass(test_embed, weights_input, biases_input, weights_output)

            # self.test_pred = self.forward_pass_sec(test_embed, weights_input, biases_input, weights_sec_input, biases_sec_input, weights_output)

            # self.test_pred = self.forward_pass_third(test_embed, weights_input, biases_input, weights_sec_input, biases_sec_input, weights_third_input, biases_third_input, weights_output)

            # intializer
            self.init = tf.global_variables_initializer()

    def train(self, sess, num_steps):
        """

        :param sess:
        :param num_steps:
        :return:
        """
        self.init.run()
        print "Initailized"

        average_loss = 0
        for step in range(num_steps):
            start = (step * Config.batch_size) % len(trainFeats)
            end = ((step + 1) * Config.batch_size) % len(trainFeats)
            if end < start:
                start -= end
                end = len(trainFeats)
            batch_inputs, batch_labels = trainFeats[start:end], trainLabels[start:end]

            feed_dict = {self.train_inputs: batch_inputs, self.train_labels: batch_labels}

            _, loss_val = sess.run([self.app, self.loss], feed_dict=feed_dict)
            average_loss += loss_val

            if step % Config.display_step == 0:
                if step > 0:
                    average_loss /= Config.display_step
                print "Average loss at step ", step, ": ", average_loss
                average_loss = 0
            if step % Config.validation_step == 0 and step != 0:
                print "\nTesting on dev set at step ", step
                predTrees = []
                for sent in devSents:
                    numTrans = parsing_system.numTransitions()

                    c = parsing_system.initialConfiguration(sent)
                    while not parsing_system.isTerminal(c):
                        feat = getFeatures(c)
                        pred = sess.run(self.test_pred, feed_dict={self.test_inputs: feat})

                        optScore = -float('inf')
                        optTrans = ""

                        for j in range(numTrans):
                            if pred[0, j] > optScore and parsing_system.canApply(c, parsing_system.transitions[j]):
                                optScore = pred[0, j]
                                optTrans = parsing_system.transitions[j]

                        c = parsing_system.apply(c, optTrans)

                    predTrees.append(c.tree)
                result = parsing_system.evaluate(devSents, predTrees, devTrees)
                print result

        print "Train Finished."

    def evaluate(self, sess, testSents):
        """

        :param sess:
        :return:
        """

        print "Starting to predict on test set"
        predTrees = []
        for sent in testSents:
            numTrans = parsing_system.numTransitions()

            c = parsing_system.initialConfiguration(sent)
            while not parsing_system.isTerminal(c):
                # feat = getFeatureArray(c)
                feat = getFeatures(c)
                pred = sess.run(self.test_pred, feed_dict={self.test_inputs: feat})

                optScore = -float('inf')
                optTrans = ""

                for j in range(numTrans):
                    if pred[0, j] > optScore and parsing_system.canApply(c, parsing_system.transitions[j]):
                        optScore = pred[0, j]
                        optTrans = parsing_system.transitions[j]

                c = parsing_system.apply(c, optTrans)

            predTrees.append(c.tree)
        print "Saved the test results."
        Util.writeConll('result_test.conll', testSents, predTrees)


    def forward_pass(self, embed, weights_input, biases_input, weights_output):
        """

        :param embed:
        :param weights:
        :param biases:
        :return:
        """
        """
        =======================================================

        Implement the forwrad pass described in
        "A Fast and Accurate Dependency Parser using Neural Networks"(2014)

        =======================================================
        embed = b X te
        weights_input = te X h
        h = b X h
        biases_input = h X 
        weights_output = h X T
        """

        '''
            Single Layer implementation of forward pass
        '''
        if Config.func == 1:
            '''
            Cubic Function as hidden layer
            '''
            print("\n\n======================")
            print("hidden size: " + str(Config.hidden_size))
            print("learning_rate: " +  str(Config.learning_rate))
            print("func: Cubic : " + str(Config.func) )
            h = tf.pow( tf.math.add( tf.matmul(embed, weights_input), biases_input), 3)
            p = tf.matmul(h,weights_output)
        elif Config.func == 2:
            '''
            Quadratic Function as hidden layer
            '''
            print("\n\n======================")
            print("hidden size: " + str(Config.hidden_size))
            print("learning_rate: " +  str(Config.learning_rate))
            print("func: quad : " + str(Config.func) )
            h = tf.pow( tf.math.add( tf.matmul(embed, weights_input), biases_input), 4)
            p = tf.matmul(h,weights_output)
        elif Config.func == 3:
            '''
            RELU Function as hidden layer
            '''
            print("\n\n======================")
            print("hidden size: " + str(Config.hidden_size))
            print("learning_rate: " +  str(Config.learning_rate))
            print("func: relu : " + str(Config.func) )
            h = tf.nn.relu( tf.math.add( tf.matmul(embed, weights_input), biases_input))
            p = tf.matmul(h,weights_output)
        elif Config.func == 4:
            '''
            Tanh Function as hidden layer
            '''
            print("\n\n======================")
            print("hidden size: " + str(Config.hidden_size))
            print("learning_rate: " +  str(Config.learning_rate))
            print("func: tanh : " + str(Config.func) )
            h = tf.nn.tanh( tf.math.add( tf.matmul(embed, weights_input), biases_input))
            p = tf.matmul(h,weights_output)
        else:
            '''
            Sigmoid Function as hidden layer
            '''
            print("\n\n======================")
            print("hidden size: " + str(Config.hidden_size))
            print("learning_rate: " +  str(Config.learning_rate))
            print("func: sigmoid : " + str(Config.func) )
            h = tf.nn.sigmoid( tf.math.add( tf.matmul(embed, weights_input), biases_input))
            p = tf.matmul(h,weights_output)
        return p
    
    def forward_pass_sec(self, embed, weights_input, biases_input, weights_input_sec, biases_input_sec, weights_output):
        """
        embed = b X te
        weights_input = te X h
        h1 = b X h
        weights_input_sec = h X h
        biases_input_sec = h X 1
        h2 = b X h
        biases_input = h X 
        weights_output = h X T


        Two Layer implementation of forward pass
        """

        if Config.func == 1:
            '''
            Cubic Function as First hidden layer 
            Cubic Function as Second hidden layer 
            '''
            print("\n\n======================")
            print("hidden size: " + str(Config.hidden_size))
            print("learning_rate: " +  str(Config.learning_rate))
            print("func: Cubic Cubic : " + str(Config.func) )
            h1 = tf.pow( tf.math.add( tf.matmul(embed, weights_input), biases_input), 3)
            h2 = tf.pow( tf.math.add( tf.matmul(h1, weights_input_sec), biases_input_sec), 3)
            p = tf.matmul(h2,weights_output)
        elif Config.func == 2:
            '''
            Cubic Function as First hidden layer 
            tanh Function as Second hidden layer 
            '''
            print("\n\n======================")
            print("hidden size: " + str(Config.hidden_size))
            print("learning_rate: " +  str(Config.learning_rate))
            print("func: Cubic Tanh : " + str(Config.func) )
            h1 = tf.pow( tf.math.add( tf.matmul(embed, weights_input), biases_input), 3)
            h2 = tf.nn.tanh( tf.math.add( tf.matmul(h1, weights_input_sec), biases_input_sec))
            p = tf.matmul(h2,weights_output)
        elif Config.func == 3:
            '''
            tanh Function as First hidden layer 
            tanh Function as Second hidden layer 
            '''
            print("\n\n======================")
            print("hidden size: " + str(Config.hidden_size))
            print("learning_rate: " +  str(Config.learning_rate))
            print("func: tanh tanh : " + str(Config.func) )
            h1 = tf.nn.tanh( tf.math.add( tf.matmul(embed, weights_input), biases_input))
            h2 = tf.nn.tanh( tf.math.add( tf.matmul(h1, weights_input_sec), biases_input_sec))
            p = tf.matmul(h2,weights_output)
        else:
            '''
            cubic Function as First hidden layer 
            relu Function as Second hidden layer 
            '''
            print("\n\n======================")
            print("hidden size: " + str(Config.hidden_size))
            print("learning_rate: " +  str(Config.learning_rate))
            print("func: Cubic relu : " + str(Config.func) )
            h1 = tf.pow( tf.math.add( tf.matmul(embed, weights_input), biases_input), 3)
            h2 = tf.nn.relu( tf.math.add( tf.matmul(h1, weights_input_sec), biases_input_sec))
            p = tf.matmul(h2,weights_output)
        return p
    
    def forward_pass_third(self, embed, weights_input, biases_input, weights_input_sec, biases_input_sec, weights_input_third, biases_input_third, weights_output):

        """
        embed = b X te
        weights_input = te X h
        h1 = b X h
        weights_input_sec = h X h
        biases_input_sec = h X 1
        h2 = b X h
        weights_input_third = h X h
        biases_input_third = h X 1
        h3 = b X h
        biases_input = h X 
        weights_output = h X T

        Three Layer implementation of forward pass
        """


        if Config.func == 1:
            print("\n\n======================")
            print("hidden size: " + str(Config.hidden_size))
            print("learning_rate: " +  str(Config.learning_rate))
            print("func: Cubic Cubic Cubic: " + str(Config.func) )
            h1 = tf.pow( tf.math.add( tf.matmul(embed, weights_input), biases_input), 3)
            h2 = tf.pow( tf.math.add( tf.matmul(h1, weights_input_sec), biases_input_sec), 3)
            h3 = tf.pow( tf.math.add( tf.matmul(h2, weights_input_third), biases_input_third), 3)
            p = tf.matmul(h3,weights_output)
        elif Config.func == 2:
            print("\n\n======================")
            print("hidden size: " + str(Config.hidden_size))
            print("learning_rate: " +  str(Config.learning_rate))
            print("func: Cubic Cubic Tanh : " + str(Config.func) )
            h1 = tf.pow( tf.math.add( tf.matmul(embed, weights_input), biases_input), 3)
            h2 = tf.pow( tf.math.add( tf.matmul(h1, weights_input_sec), biases_input_sec), 3)
            h3 = tf.nn.tanh( tf.math.add( tf.matmul(h2, weights_input_third), biases_input_third))
            p = tf.matmul(h3,weights_output)
        elif Config.func == 3:
            print("\n\n======================")
            print("hidden size: " + str(Config.hidden_size))
            print("learning_rate: " +  str(Config.learning_rate))
            print("func: tanh tanh tanh : " + str(Config.func) )
            h1 = tf.nn.tanh( tf.math.add( tf.matmul(embed, weights_input), biases_input))
            h2 = tf.nn.tanh( tf.math.add( tf.matmul(h1, weights_input_sec), biases_input_sec))
            h3 = tf.nn.tanh( tf.math.add( tf.matmul(h2, weights_input_third), biases_input_third))
            p = tf.matmul(h3,weights_output)
        else:
            print("\n\n======================")
            print("hidden size: " + str(Config.hidden_size))
            print("learning_rate: " +  str(Config.learning_rate))
            print("func: Cubic Cubic relu : " + str(Config.func) )
            h1 = tf.pow( tf.math.add( tf.matmul(embed, weights_input), biases_input), 3)
            h2 = tf.pow( tf.math.add( tf.matmul(h1, weights_input_sec), biases_input_sec), 3)
            h3 = tf.nn.relu( tf.math.add( tf.matmul(h2, weights_input_third), biases_input_third))
            p = tf.matmul(h3,weights_output)
        return p


def genDictionaries(sents, trees):
    word = []
    pos = []
    label = []
    for s in sents:
        for token in s:
            word.append(token['word'])
            pos.append(token['POS'])

    rootLabel = None
    for tree in trees:
        for k in range(1, tree.n + 1):
            if tree.getHead(k) == 0:
                rootLabel = tree.getLabel(k)
            else:
                label.append(tree.getLabel(k))

    if rootLabel in label:
        label.remove(rootLabel)

    index = 0
    wordCount = [Config.UNKNOWN, Config.NULL, Config.ROOT]
    wordCount.extend(collections.Counter(word))
    for word in wordCount:
        wordDict[word] = index
        index += 1

    posCount = [Config.UNKNOWN, Config.NULL, Config.ROOT]
    posCount.extend(collections.Counter(pos))
    for pos in posCount:
        posDict[pos] = index
        index += 1

    labelCount = [Config.NULL, rootLabel]
    labelCount.extend(collections.Counter(label))
    for label in labelCount:
        labelDict[label] = index
        index += 1

    return wordDict, posDict, labelDict


def getWordID(s):
    if s in wordDict:
        return wordDict[s]
    else:
        return wordDict[Config.UNKNOWN]


def getPosID(s):
    if s in posDict:
        return posDict[s]
    else:
        return posDict[Config.UNKNOWN]


def getLabelID(s):
    if s in labelDict:
        return labelDict[s]
    else:
        return labelDict[Config.UNKNOWN]


def getFeatures(c):

    """
    =================================================================

    Implement feature extraction described in
    "A Fast and Accurate Dependency Parser using Neural Networks"(2014)

    =================================================================
    """
    features = []
    token_index = []

    """
    First Three words on the stack
    """
    s1 = c.getStack(0)
    s2 = c.getStack(1)
    s3 = c.getStack(2)

    # features.extend([self.getWordID(c.getWord(s1)), self.getWordID(c.getWord(s2)), self.getWordID(c.getWord(s3))])

    """
    First Three words on the buffer
    """

    b1 = c.getBuffer(0)
    b2 = c.getBuffer(1)
    b3 = c.getBuffer(2)

    # features.extend([self.getWordID(getWord(b1)), self.getWordID(getWord(b2)), self.getWordID(getWord(b3))])

    """
    1st leftmost child of s1
    1st rightmost child of s1

    2nd leftmost child of s1
    2nd rightmost child of s1

    """

    s1lc1 = c.getLeftChild(s1,1)
    s1rc1 = c.getRightChild(s1,1)
    s1lc2 = c.getLeftChild(s1,2)
    s1rc2 = c.getRightChild(s1,2)


    """
    1st leftmost child of s2
    1st rightmost child of s2

    2nd leftmost child of s2
    2nd rightmost child of s2

    """

    s2lc1 = c.getLeftChild(s2,1)
    s2rc1 = c.getRightChild(s2,1)
    s2lc2 = c.getLeftChild(s2,2)
    s2rc2 = c.getRightChild(s2,2)

    """
    1st leftmost child of 1st leftmost child of s1
    1st rightmost child of 1st rightmost child of s1

    1st leftmost child of 1st leftmost child of s2
    1st rightmost child of 1st rightmost child of s2

    """

    s1lc1lc1 = c.getLeftChild(s1lc1,1)
    s1rc1rc1 = c.getRightChild(s1rc1,1)

    s2lc1lc1 = c.getLeftChild(s2lc1,1)
    s2rc1rc1 = c.getRightChild(s2rc1,1)


    #Concatenate all the words
    token_index.extend([s1, s2, s3, b1, b2, b3, s1lc1, s1rc1, s1lc2, s1rc2, s2lc1, s2rc1, s2lc2, s2rc2, s1lc1lc1, s1rc1rc1, s2lc1lc1, s2rc1rc1])


    #Get the word id of all words
    for w in token_index:
        features.extend( [ getWordID(c.getWord(w)) ] )

    #Get the post id of all words
    for w in token_index:
        features.extend( [ getPosID(c.getPOS(w)) ] )

    #Get the label id of last 12 words
    #Total = 18+18+12 = 48
    for w in range(6,len(token_index)):
        features.extend( [ getLabelID(c.getLabel( token_index[w] )) ] )
    return features


def genTrainExamples(sents, trees):
    numTrans = parsing_system.numTransitions()

    features = []
    labels = []
    pbar = ProgressBar()
    for i in pbar(range(len(sents))):
        if trees[i].isProjective():
            c = parsing_system.initialConfiguration(sents[i])

            while not parsing_system.isTerminal(c):
                oracle = parsing_system.getOracle(c, trees[i])
                feat = getFeatures(c)
                label = []
                for j in range(numTrans):
                    t = parsing_system.transitions[j]
                    if t == oracle:
                        label.append(1.)
                    elif parsing_system.canApply(c, t):
                        label.append(0.)
                    else:
                        label.append(-1.)

                if 1.0 not in label:
                    print i, label
                features.append(feat)
                labels.append(label)
                c = parsing_system.apply(c, oracle)
    return features, labels


def load_embeddings(filename, wordDict, posDict, labelDict):
    dictionary, word_embeds = pickle.load(open(filename, 'rb'))

    embedding_array = np.zeros((len(wordDict) + len(posDict) + len(labelDict), Config.embedding_size))
    knownWords = wordDict.keys()
    foundEmbed = 0
    for i in range(len(embedding_array)):
        index = -1
        if i < len(knownWords):
            w = knownWords[i]
            if w in dictionary:
                index = dictionary[w]
            elif w.lower() in dictionary:
                index = dictionary[w.lower()]
        if index >= 0:
            foundEmbed += 1
            embedding_array[i] = word_embeds[index]
        else:
            embedding_array[i] = np.random.rand(Config.embedding_size) * 0.02 - 0.01
    print "Found embeddings: ", foundEmbed, "/", len(knownWords)

    return embedding_array


if __name__ == '__main__':

    wordDict = {}
    posDict = {}
    labelDict = {}
    parsing_system = None

    trainSents, trainTrees = Util.loadConll('train.conll')
    devSents, devTrees = Util.loadConll('dev.conll')
    testSents, _ = Util.loadConll('test.conll')
    genDictionaries(trainSents, trainTrees)

    embedding_filename = 'word2vec.model'

    embedding_array = load_embeddings(embedding_filename, wordDict, posDict, labelDict)

    labelInfo = []
    for idx in np.argsort(labelDict.values()):
        labelInfo.append(labelDict.keys()[idx])
    parsing_system = ParsingSystem(labelInfo[1:])
    print parsing_system.rootLabel

    print "Generating Traning Examples"  
    trainFeats, trainLabels = genTrainExamples(trainSents, trainTrees)
    print "Done."

    #hidden layer size
    # lhs = [200, 400, 600, 1000, 3000]
    lhs = [3000]

    #Learning rate
    # llr = [0.05, 0.1, 0.2, 0.3, 0.5]
    llr = [0.5]
    
    #Function being used in forward pass
    # lfunc = [2, 3, 4, 5]
    lfunc = [4]

    #Number of iterations
    # liter = [20001]
    # for iter in liter:

    
    for func in lfunc:
        for hs in lhs:
            for lr in llr:
                """
                Changing the hyper parameter
                hidden_size = hs
                learning_rate = lr
                func = func

                """
                Config.hidden_size = hs
                Config.learning_rate = lr
                Config.func = func
                Config.max_iter = 20001

                # Build the graph model
                graph = tf.Graph()
                model = DependencyParserModel(graph, embedding_array, Config)

                num_steps = Config.max_iter
                with tf.Session(graph=graph) as sess:

                    model.train(sess, num_steps)

                    model.evaluate(sess, testSents)

