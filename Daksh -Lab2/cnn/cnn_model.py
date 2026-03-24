import numpy as np

# =============================================================================
# ACTIVATION FUNCTIONS
# =============================================================================

def relu(x):
    return np.maximum(0, x)


def relu_backward(dout, x):
    return dout * (x > 0)


def softmax(x):
    shifted = x - np.max(x, axis=1, keepdims=True)
    exp_x = np.exp(shifted)
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)


# =============================================================================
# LOSS FUNCTION
# =============================================================================

def cross_entropy_loss(y_pred, y_true):
    N = y_pred.shape[0]
    y_pred_clipped = np.clip(y_pred, 1e-12, 1.0)

    correct_log_probs = -np.log(y_pred_clipped[np.arange(N), y_true])
    loss = np.sum(correct_log_probs) / N

    grad = y_pred_clipped.copy()
    grad[np.arange(N), y_true] -= 1
    grad /= N

    return loss, grad


# =============================================================================
# CONVOLUTIONAL LAYER
# =============================================================================

class ConvLayer:
    def __init__(self, in_channels, num_filters, kernel_size, stride=1, padding=0):
        self.in_channels = in_channels
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        fan_in = in_channels * kernel_size * kernel_size
        self.W = np.random.randn(num_filters, in_channels, kernel_size, kernel_size) * np.sqrt(2.0 / fan_in)
        self.b = np.zeros(num_filters)

        self.vW = np.zeros_like(self.W)
        self.vb = np.zeros_like(self.b)

        self.x_padded = None
        self.x_shape = None
        self.dW = None
        self.db = None

    def forward(self, x):
        N, C, H, W = x.shape
        K, S, P = self.kernel_size, self.stride, self.padding

        if P > 0:
            x_padded = np.pad(x, ((0, 0), (0, 0), (P, P), (P, P)), mode='constant')
        else:
            x_padded = x

        H_out = (H + 2 * P - K) // S + 1
        W_out = (W + 2 * P - K) // S + 1

        out = np.zeros((N, self.num_filters, H_out, W_out))

        for f in range(self.num_filters):
            for i in range(H_out):
                for j in range(W_out):
                    patch = x_padded[:, :, i*S:i*S+K, j*S:j*S+K]
                    out[:, f, i, j] = np.sum(patch * self.W[f], axis=(1, 2, 3)) + self.b[f]

        self.x_padded = x_padded
        self.x_shape = x.shape
        return out

    def backward(self, dout):
        N, C, H, W = self.x_shape
        K, S, P = self.kernel_size, self.stride, self.padding
        _, _, H_out, W_out = dout.shape

        self.dW = np.zeros_like(self.W)
        self.db = np.sum(dout, axis=(0, 2, 3))
        dx_padded = np.zeros_like(self.x_padded)

        for f in range(self.num_filters):
            for i in range(H_out):
                for j in range(W_out):
                    patch = self.x_padded[:, :, i*S:i*S+K, j*S:j*S+K]
                    dout_ij = dout[:, f, i, j]

                    self.dW[f] += np.sum(dout_ij[:, None, None, None] * patch, axis=0)
                    dx_padded[:, :, i*S:i*S+K, j*S:j*S+K] += (
                        dout_ij[:, None, None, None] * self.W[f]
                    )

        if P > 0:
            dx = dx_padded[:, :, P:-P, P:-P]
        else:
            dx = dx_padded

        return dx

    def update(self, lr, momentum=0.9):
        dW = np.clip(self.dW, -5, 5)
        db = np.clip(self.db, -5, 5)

        self.vW = momentum * self.vW - lr * dW
        self.vb = momentum * self.vb - lr * db
        self.W += self.vW
        self.b += self.vb


# =============================================================================
# MAX POOLING LAYER
# =============================================================================

class MaxPoolLayer:
    def __init__(self, pool_size=2, stride=2):
        self.pool_size = pool_size
        self.stride = stride
        self.x = None

    def forward(self, x):
        N, C, H, W = x.shape
        P, S = self.pool_size, self.stride
        H_out = (H - P) // S + 1
        W_out = (W - P) // S + 1

        out = np.zeros((N, C, H_out, W_out))

        for i in range(H_out):
            for j in range(W_out):
                window = x[:, :, i*S:i*S+P, j*S:j*S+P]
                out[:, :, i, j] = np.max(window, axis=(2, 3))

        self.x = x
        return out

    def backward(self, dout):
        x, P, S = self.x, self.pool_size, self.stride
        N, C, H, W = x.shape
        H_out = (H - P) // S + 1
        W_out = (W - P) // S + 1
        dx = np.zeros_like(x)

        for i in range(H_out):
            for j in range(W_out):
                window = x[:, :, i*S:i*S+P, j*S:j*S+P]
                max_v = np.max(window, axis=(2, 3), keepdims=True)
                mask = (window == max_v)
                tie_count = np.sum(mask, axis=(2, 3), keepdims=True)

                dx[:, :, i*S:i*S+P, j*S:j*S+P] += (
                    mask * dout[:, :, i, j][:, :, None, None] / tie_count
                )

        return dx


# =============================================================================
# FULLY CONNECTED LAYER
# =============================================================================

class FCLayer:
    def __init__(self, in_features, out_features):
        self.in_features = in_features
        self.out_features = out_features

        self.W = np.random.randn(in_features, out_features) * np.sqrt(2.0 / in_features)
        self.b = np.zeros(out_features)

        self.vW = np.zeros_like(self.W)
        self.vb = np.zeros_like(self.b)

        self.x = None
        self.dW = None
        self.db = None

    def forward(self, x):
        self.x = x
        return x @ self.W + self.b

    def backward(self, dout):
        self.dW = self.x.T @ dout
        self.db = np.sum(dout, axis=0)
        dx = dout @ self.W.T
        return dx

    def update(self, lr, momentum=0.9):
        dW = np.clip(self.dW, -5, 5)
        db = np.clip(self.db, -5, 5)

        self.vW = momentum * self.vW - lr * dW
        self.vb = momentum * self.vb - lr * db
        self.W += self.vW
        self.b += self.vb


# =============================================================================
# CNN MODEL
# =============================================================================

class CNN:
    def __init__(self):
        self.conv1 = ConvLayer(1, 8, 3)
        self.pool1 = MaxPoolLayer(2, 2)
        self.conv2 = ConvLayer(8, 16, 3)
        self.pool2 = MaxPoolLayer(2, 2)
        self.fc1 = FCLayer(400, 128)
        self.fc2 = FCLayer(128, 10)

        self.relu1_input = None
        self.relu2_input = None
        self.relu3_input = None
        self.flat_shape = None

    def forward(self, x):
        x = self.conv1.forward(x)
        self.relu1_input = x
        x = relu(x)
        x = self.pool1.forward(x)

        x = self.conv2.forward(x)
        self.relu2_input = x
        x = relu(x)
        x = self.pool2.forward(x)

        self.flat_shape = x.shape
        x = x.reshape(x.shape[0], -1)

        x = self.fc1.forward(x)
        self.relu3_input = x
        x = relu(x)

        x = self.fc2.forward(x)
        return softmax(x)

    def backward(self, grad):
        grad = self.fc2.backward(grad)
        grad = relu_backward(grad, self.relu3_input)
        grad = self.fc1.backward(grad)

        grad = grad.reshape(self.flat_shape)

        grad = self.pool2.backward(grad)
        grad = relu_backward(grad, self.relu2_input)
        grad = self.conv2.backward(grad)

        grad = self.pool1.backward(grad)
        grad = relu_backward(grad, self.relu1_input)
        grad = self.conv1.backward(grad)

    def update(self, lr, momentum=0.9):
        self.conv1.update(lr, momentum)
        self.conv2.update(lr, momentum)
        self.fc1.update(lr, momentum)
        self.fc2.update(lr, momentum)

    def predict(self, x):
        probs = self.forward(x)
        return np.argmax(probs, axis=1)