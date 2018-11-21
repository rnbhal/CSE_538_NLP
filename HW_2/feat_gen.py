#!/bin/python

adjectives = ["other","new","good","high","old","great","big","American","small","large","national","young","different","black","long","little","important","political","bad","white","real","best","right","social","only","public","sure","low","early","able","human","local","late","hard","major","better","economic","strong","possible","whole","free","military","TRUE","federal","international","full","special","easy","clear","recent","certain","personal","open","red","difficult","available","likely","short","single","medical","current","wrong","private","past","foreign","fine","common","poor","natural","significant","similar","hot","dead","central","happy","serious","ready","simple","left","physical","general","environmental","financial","blue","democratic","dark","various","entire","close","legal","religious","cold","final","main","green","nice","huge","popular","traditional","cultural","able","bad","best","better","big","black","certain","clear","different","early","easy","economic","federal","free","full","good","great","hard","high","human","important","international","large","late","little","local","long","low","major","military","national","new","old","only","other","political","possible","public","real","recent","right","small","social","special","strong","sure","true","white","whole","young"]
verb = ["ask","be","become","begin","call","can","come","could","do","feel","find","get","give","go","have","hear","help","keep","know","leave","let","like","live","look","make","may","mean","might","move","need","play","put","run","say","see","seem","should","show","start","take","talk","tell","think","try","turn","use","want","will","work","would"]
preposition = ["up","down","beneath","of","in","to","for","with","on","at","from","by","about","as","into","like","through","after","over","between","out","against","during","without","before","under","around","among"]
noun = ["America", "area","book","business","case","child","company","country","day","eye","fact","family","government","group","hand","home","job","life","lot","man","money","month","mother","Mr","night","number","part","people","place","point","problem","program","question","right","room","school","state","story","student","study","system","thing","time","water","way","week","woman","word","work","world","year"]

def preprocess_corpus(train_sents):
    """Use the sentences to do whatever preprocessing you think is suitable,
    such as counts, keeping track of rare features/words to remove, matches to lexicons,
    loading files, and so on. Avoid doing any of this in token2features, since
    that will be called on every token of every sentence.

    Of course, this is an optional function.

    Note that you can also call token2features here to aggregate feature counts, etc.
    """
    pass

def token2features(sent, i, add_neighs = True):
    """Compute the features of a token.

    All the features are boolean, i.e. they appear or they do not. For the token,
    you have to return a set of strings that represent the features that *fire*
    for the token. See the code below.

    The token is at position i, and the rest of the sentence is provided as well.
    Try to make this efficient, since it is called on every token.

    One thing to note is that it is only called once per token, i.e. we do not call
    this function in the inner loops of training. So if your training is slow, it's
    not because of how long it's taking to run this code. That said, if your number
    of features is quite large, that will cause slowdowns for sure.

    add_neighs is a parameter that allows us to use this function itself in order to
    recursively add the same features, as computed for the neighbors. Of course, we do
    not want to recurse on the neighbors again, and then it is set to False (see code).
    """
    # global adjectives
    ftrs = []
    # bias
    ftrs.append("BIAS")
    # position features
    if i == 0:
        ftrs.append("SENT_BEGIN")
    if i == len(sent)-1:
        ftrs.append("SENT_END")

    # the word itself
    word = unicode(sent[i])
    ftrs.append("WORD=" + word)
    ftrs.append("LCASE=" + word.lower())
    # some features of the word
    if word.isalnum():
        ftrs.append("IS_ALNUM")
    if word.isnumeric():
        ftrs.append("IS_NUMERIC")
    if word.isdigit():
        ftrs.append("IS_DIGIT")
    if word.isupper():
        ftrs.append("IS_UPPER")
    if word.islower():
        ftrs.append("IS_LOWER")
    
    if word.istitle():
        ftrs.append("IS_TITLE")

    if word.__contains__("."):
        ftrs.append("HAS_PERIOD")
    if word.__contains__(","):
        ftrs.append("HAS_PUNC")
    if word.__contains__("'"):
        ftrs.append("HAS_PUNC2")
    if adjectives.__contains__(word):
        ftrs.append("HAS_ADJECTIVE")
    if verb.__contains__(word):
        ftrs.append("HAS_VERB")
    if preposition.__contains__(word):
        ftrs.append("HAS_PREPOSITION")
    if noun.__contains__(word):
        ftrs.append("HAS_NOUN")
    if word.__contains__("?"):
        ftrs.append("HAS_QUESTIONS")
    # if word.__contains__("/"):
    #     ftrs.append("HAS_/")
    # if word.__contains__("\\"):
    #     ftrs.append("HAS_\\")
    # if word.__contains__("http"):
    #     ftrs.append("HAS_HTTP")
    # if word.__contains__("&"):
    #     ftrs.append("HAS_&")
    # if word.__contains__("%"):
    #     ftrs.append("HAS_%")
    if word.__contains__("-"):
        ftrs.append("HAS_HYPHEN")
    if word.__contains__("@"):
        ftrs.append("HAS_@")
    if word.__contains__("!"):
        ftrs.append("HAS_!")
    if word.__contains__(":"):
        ftrs.append("HAS_:")
    if word.__contains__(")"):
        ftrs.append("HAS_)")
    if word.__contains__("|"):
        ftrs.append("HAS_|")
    if word.__contains__("$"):
        ftrs.append("HAS_$")
    if word.__contains__("#"):
        ftrs.append("HAS_#")
    if word.__contains__("com") or word.__contains__("COM"):
        ftrs.append("HAS_COM")

    if word.endswith("ing"):
        ftrs.append("SUFFIX_ING")
    if word.endswith("ment"):
        ftrs.append("SUFFIX_MENT")
    if word.endswith("ion"):
        ftrs.append("SUFFIX_ION")
    if word.endswith("ful"):
        ftrs.append("SUFFIX_FUL")
    if word.endswith("ness"):
        ftrs.append("SUFFIX_NESS")
    if word.endswith("al"):
        ftrs.append("SUFFIX_AL")
    if word.endswith("ify"):
        ftrs.append("SUFFIX_IFY")
    if word.endswith("ate"):
        ftrs.append("SUFFIX_ATE")
    if word.endswith("ly"):
        ftrs.append("SUFFIX_LY")
    if word.endswith("ward"):
        ftrs.append("SUFFIX_WARD")
    if word.endswith("lity"):
        ftrs.append("SUFFIX_LITY")
    if word.endswith("ism"):
        ftrs.append("SUFFIX_ISM")
    if word.endswith("ist"):
        ftrs.append("SUFFIX_IST")
    
    
    if word.startswith("de"):
        ftrs.append("PREFIX_DE")
    if word.startswith("dis"):
        ftrs.append("PREFIX_DIS")
    if word.startswith("il"):
        ftrs.append("PREFIX_IL")
    if word.startswith("ex"):
        ftrs.append("PREFIX_EX")
    if word.startswith("in"):
        ftrs.append("PREFIX_IN")
    if word.startswith("re"):
        ftrs.append("PREFIX_RE")
    if word.startswith("un"):
        ftrs.append("PREFIX_UN")
    if word.startswith("im"):
        ftrs.append("PREFIX_IM")

    
    if len(ftrs) <= 6 :
        ftrs.append("NOT_CLASSIFIED")



    # previous/next word feats
    if add_neighs:
        if i > 0:
            for pf in token2features(sent, i-1, add_neighs = False):
                ftrs.append("PREV_" + pf)
        if i < len(sent)-1:
            for pf in token2features(sent, i+1, add_neighs = False):
                ftrs.append("NEXT_" + pf)

    # return it!
    return ftrs

if __name__ == "__main__":
    sents = [ [ "I", "aaaahhh", "food", "cooking" ]]
    preprocess_corpus(sents)
    for sent in sents:
        for i in xrange(len(sent)):
            print sent[i], ":", token2features(sent, i)
