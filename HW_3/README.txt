te = Config.n_Tokens*Config.embedding_size
h = Config.hidden_size
b = Config.batch_size
T = numTrans



Forward Pass Implementation Details :

==============================================SINGLE LAYER====================================================

For Single Layer Forward Pass :

Hyper Parameter

Config.func = 1 ==> Cubic Function as hidden layer
Config.func = 2 ==> Quadratic Function as hidden layer
Config.func = 3 ==> RELU Function as hidden layer
Config.func = 4 ==> tanh Function as hidden layer
Config.func = other ==> Sigmoid Function as hidden layer
 

weights_input = First Layer weight
biases_input = First Layer bias


def forward_pass(self, embed, weights_input, biases_input, weights_output):
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



self.prediction = self.forward_pass(train_embed, weights_input, biases_input, weights_output)



==============================================END =========================================================










==============================================TWO LAYER====================================================


For Two Layer Forward Pass:




Hyperparameter
Config.func = 1 ==> Cubic Function as First hidden layer 
            Cubic Function as Second hidden layer 
Config.func = 2 ==> Cubic Function as First hidden layer 
            tanh Function as Second hidden layer 
Config.func = 3 ==> tanh Function as First hidden layer 
            tanh Function as Second hidden layer 
Config.func = other ==> cubic Function as First hidden layer 
            relu Function as Second hidden layer 


weights_sec_input = Second Layer weight
biases_input_sec = Second Layer bias

def forward_pass_sec(self, embed, weights_input, biases_input, weights_input_sec, biases_input_sec,weights_output):

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



self.prediction = self.forward_pass_sec(train_embed, weights_input, biases_input, weights_sec_input, biases_sec_input, weights_output)


==============================================END ===========================================================











==============================================THREE LAYER====================================================


Three Layer implementation of forward pass


Config.func = 1 ==> Cubic Function as First hidden layer 
            Cubic Function as Second hidden layer
	    Cubic Function as Third hidden layer  
Config.func = 2 ==> Cubic Function as First hidden layer 
            Cubic Function as Second hidden layer
	    tanh Function as Third hidden layer 
Config.func = 3 ==> tanh Function as First hidden layer 
            tanh Function as Second hidden layer
	    tanh Function as Third hidden layer 
Config.func = other ==> Cubic Function as First hidden layer 
            Cubic Function as Second hidden layer
	    Relu Function as Third hidden layer 


weights_sec_input = Second Layer weight
biases_input_sec = Second Layer bias

weights_input_third = Third Layer weight
biases_input_third = Third Layer bias


def forward_pass_third(self, embed, weights_input, biases_input, weights_input_sec, biases_input_sec, weights_input_third, biases_input_third, weights_output):
For Three Layer Forward Pass:

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



self.prediction = self.forward_pass_sec(train_embed, weights_input, biases_input, weights_sec_input, biases_sec_input, weights_third_input, biases_third_input, weights_output)


==============================================END====================================================

