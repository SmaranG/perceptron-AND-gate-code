import numpy as np

class Perceptron:
    def __init__(self, lr=0.1, th=0.0):
        self.lr = lr
        self.th = th

    def step(self, val):
        return 1 if val >= self.th else 0

    def predict(self, x_in):
        weighted_sum = np.dot(x_in, self.weights) + self.bias
        return self.step(weighted_sum)

    def fit(self, x, y):
        self.x = x
        self.y = y

        self.weights = np.random.rand(self.x.shape[1])
        self.bias = np.random.rand()

        is_iterating = True
        while is_iterating:
            is_iterating = False
            for x_in, y_actual in zip(self.x, self.y):
                y_pred = self.predict(x_in)
                error = y_actual - y_pred

                if error != 0:
                    # Update weights
                    self.weights += self.lr * error * x_in
                    # Update bias
                    self.bias += self.lr * error
                    is_iterating = True

        return self.weights, self.bias

x = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])

y = np.array([0, 0, 0, 1])

and_gate_p = Perceptron()
f_weights, f_bias = and_gate_p.fit(x, y)

print("Final Weights:", ", ".join(str(w) for w in f_weights))
print("Final Bias:", f_bias)

print("\nPredictions:")
for x_in in x:
    print(f"{x_in} -> {and_gate_p.predict(x_in)}")
