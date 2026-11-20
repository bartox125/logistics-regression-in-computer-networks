import numpy as np

def sigmoid(z):
    z_clip=np.clip(z, -500,500)
    return 1.0/(1.0+np.exp(-z_clip))

def compute_loss(y, y_hat):
    eps = 1e-15
    y_hat = np.clip(y_hat,eps,1-eps)
    return -np.mean(y*np.log(y_hat)+(1-y)*np.log(1-y_hat))

def predict(x,weights,b):
    z=x @ weights +b
    return sigmoid(z)

def compare(x, y, weights, b, threshold=0.5):
    y = np.asarray(y).reshape(-1)
    pred = (predict(np.asarray(x), weights, b) >= threshold).astype(int)

    real_bad = int(np.sum(y == 1))
    real_safe = int(np.sum(y == 0))
    bad_detected = int(np.sum((pred == 1) & (y == 1)))
    safe_detected = int(np.sum((pred == 0) & (y == 0)))

    print(f"Bad packets: {real_bad}, detected: {bad_detected}")
    print(f"Normal packets: {real_safe}, detected: {safe_detected}")
    print(f"Accuracy: {np.mean(pred == y):.4f}")

    tp = bad_detected
    fp = real_safe - safe_detected 
    fn = real_bad - bad_detected  
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}")

def train(x,y,lr,epochs):
    y=np.asarray(y).reshape(-1)
    x=np.asarray(x,dtype=float)
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

        if i %100==0:
            y_hat_new = sigmoid(x @ weights + b)
            loss = compute_loss(y, y_hat_new)
            print(f"Epoch: {i}, Loss: {loss}")

    return weights, b
