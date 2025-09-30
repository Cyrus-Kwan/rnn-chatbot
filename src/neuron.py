from typing import Callable
from math import exp, tanh

class Neuron():
    def __init__(self, inputs, weights, active, learn):
        self.inputs:list[float]     = inputs
        self.weights:list[float]    = weights
        self.active:Callable    = active
        self.learn:Callable     = learn
        self.prev_delta:list    = [0.0] * len(weights)  # added for momentum

    def scalar(self) -> float:
        '''
        The dot product of input and weight vectors returned as a scalar.
        '''
        result:float    = 0

        for i, w in zip(self.inputs, self.weights):
            result  += i * w

        return result

class Activation():
    @staticmethod
    def sign_zero(self:Neuron) -> int:
        if self.scalar() >= 0:
            return 1
        else:
            return 0
        
    @staticmethod
    def sign_one(self:Neuron) -> int:
        if self.scalar() >= 0:
            return 1
        else:
            return -1
        
    @staticmethod
    def logistic(self:Neuron) -> float:
        return 1/(1+exp(-self.scalar()))
    
    @staticmethod
    def bipolar(self:Neuron) -> float:
        return (2/(1+exp(-self.scalar()))) - 1
    
    @staticmethod
    def hyper_log(self:Neuron) -> float:
        return tanh(self.scalar())
    
class LearningRule():
    @staticmethod
    def hebbian(self:Neuron, c:float) -> None:
        '''
        r: learning signal
        x: input vector
        c: learning constant
        '''
        r   = self.active(self)
        x   = self.inputs

        for i in range(len(self.weights)):
            self.weights[i] += c * r * x[i]

        return
    
    @staticmethod
    def discrete(self:Neuron, c:float, d:float) -> None:
        '''
        r: learning signal error (r(k) = e(k))
        x: input vector
        c: learning constant
        d: desired output
        z: output vector
        '''
        z   = self.active(self)
        r   = d - z
        x   = self.inputs

        for i in range(len(self.weights)):
            self.weights[i] += c * r * x[i]

        return
    
    @staticmethod
    def gradient(self:Neuron, c:float, d:float) -> None:
        '''
        r: learning signal
        x: input vector
        c: learning constant
        d: desired output
        z: output vector
        u: derivative of activation function f(v)'
        e: error function (d-z)*u
        '''
        z   = self.active(self)
        e   = d - z
        u   = None

        if self.active == Activation.logistic:
            u   = Derivative.logistic
        elif self.active == Activation.bipolar:
            u   = Derivative.bipolar

        r   = e * u(z)
        x   = self.inputs

        for i in range(len(self.weights)):
            self.weights[i] += c * r * x[i]

        return

class Derivative():
    @staticmethod
    def logistic(z:float) -> float:
        return z * (1 - z)
    
    @staticmethod
    def bipolar(z:float) -> float:
        return 0.5 * (1 - z**2)

def main():
    return

if __name__ == "__main__":
    main()