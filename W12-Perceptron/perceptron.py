# %%
"""
<h2>Perceptron implementation</h2>
by Vip Lab 116 - EE Dept. NCNU TW
"""

# %%
"""
numpy lets us create vectors, and gives us both linear algebra functions and python list-like methods to use with it. We access its functions by calling them on np.
"""

# %%
import numpy as np

# %%
"""
Here, we’re creating a new class Perceptron. This will, among other things, allow us to maintain state in order to use our perceptron after it has learned and assigned values to its weights.
1. __init__ function <br/>
The <b>no_of_inputs</b> is used to determine how many weights we need to learn. <br/>
The <b>threshold</b>, is the number of epochs we’ll allow our learning algorithm to iterate through before ending, and it’s defaulted to 100. <br/>
The <b>learning_rate</b> is used to determine the magnitude of change for our weights during each step through our training data, and is defaulted to 0.01. <br/>
Initialize a <b>weight</b> vector with an n-number of 0’s. <br/>

2. __predict__ method <br/>
f(x) = 1 if w · x + b > 0 : 0 otherwise <br/>
dot product function: np.dot(a, b) == a · b <br/>
f the summation from above is greater than 0, we store 1 in the variable activation, otherwise, activation = 0, then we return that value.<br/>

3. __train__ method: which takes two arguments: training_inputs and labels <br/>
The <b>labels</b> is expected to be a numpy array of expected output values for each of the corresponding inputs in the <b>training_inputs</b> list.<br/>



"""

# %%
class Perceptron(object):
    def __init__(self, no_of_inputs, threshold=100, learning_rate=0.01):
        self.threshold = threshold
        self.learning_rate = learning_rate
        self.weights = np.zeros(no_of_inputs + 1)
        
    def predict(self, inputs):
        summation = np.dot(inputs, self.weights[1:]) + self.weights[0]
        if summation > 0:
          activation = 1
        else:
          activation = 0            
        return activation
    
    def train(self, training_inputs, labels):
        for _ in range(self.threshold):
            for inputs, label in zip(training_inputs, labels):
                prediction = self.predict(inputs)
                self.weights[1:] += self.learning_rate * (label - prediction) * inputs
                self.weights[0] += self.learning_rate * (label - prediction)