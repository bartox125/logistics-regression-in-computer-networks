import numpy as np

def sigmoid(z):
    z_clip=np.clip(z, -500,500)
    return 1.0/(1.0+np.exp(-z_clip))

def compute_loss(y, y_hat):
    eps = 1e-15
    y_hat = np.clip(y_hat,eps,1-eps)
    return -np.mean(y*np.log(y_hat)+(1-y)*np.log(1-y_hat))


def train(x,y,lr,epochs):
    m, count_of_attributes=x.shape

    weights = np.zeros(count_of_attributes, dtype=float)
    b=0.0

    for i in range(epochs):
        z=x @ weights + b
        y_hat = sigmoid(z)

        dw = (1.0/m) * (x.T @ (y_hat-y))
        db = (1.0/m)*np.sum(y_hat-y)
        weights-=lr *dw
        b-=lr*db
    return weights, b


