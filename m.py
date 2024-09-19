import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 

# تابع سیگموئید دو قطبی

def sigmoid(x):
    return 2 / (1 + np.exp(-x)) - 1

# مشتق تابع سیگموئید دو قطبی

def sigmoid_derivative(x):
    return 0.5 * (1 - sigmoid(x)) * (1 + sigmoid(x))

# تابع آموزش شبکه

def train(X, y, hidden_neurons, learning_rate, epochs):
    input_neurons = 63
    output_neurons = 26

    # وزن‌ها به صورت تصادفی مقداردهی اولیه می‌شوند
    np.random.seed(1)
    weights_input_hidden = np.random.uniform(-1,
                                             1, (input_neurons, hidden_neurons))
    weights_hidden_output = np.random.uniform(-1,
                                              1, (hidden_neurons, output_neurons))

    for epoch in range(epochs):
        # فیدفوروارد
        hidden_layer_input = np.dot(X, weights_input_hidden)
        hidden_layer_output = sigmoid(hidden_layer_input)

        output_layer_input = np.dot(hidden_layer_output, weights_hidden_output)
        output = sigmoid(output_layer_input)

        # محاسبه خطا
        error = y - output

        # برگشت به عقب و به‌روزرسانی وزن‌ها
        output_delta = error * sigmoid_derivative(output_layer_input)
        hidden_error = output_delta.dot(weights_hidden_output.T)
        hidden_delta = hidden_error * sigmoid_derivative(hidden_layer_input)

        weights_hidden_output += hidden_layer_output.T.dot(
            output_delta) * learning_rate
        weights_input_hidden += X.T.dot(hidden_delta) * learning_rate

    return weights_input_hidden, weights_hidden_output


def generate_data(name):
    df = pd.read_csv(name, sep=',', header=None)
    df.iloc[:, 63] = pd.Categorical(df.iloc[:, 63]).codes
    X = df.drop(63, axis=1).to_numpy()
    Y = df[63].to_numpy(dtype=np.int16)
    label = np.eye(26)[Y]
    return X, label


X_train, y_train = generate_data('train/alphabet_dataset.txt')
X_test10, y_test10 = generate_data('test/letter10error.txt')
X_test15, y_test15 = generate_data('test/letter15error.txt')
X_test20, y_test20 = generate_data('test/letter20error.txt')


result_lr = []

for lr in range(1, 11, 1):
    
    weights_input_hidden, weights_hidden_output = train(X_train, y_train, 20, lr/10, 900)

    # تست شبکه
    hidden_layer_input_test = np.dot(X_test10, weights_input_hidden)
    hidden_layer_output_test = sigmoid(hidden_layer_input_test)
    output_layer_input_test = np.dot(
        hidden_layer_output_test, weights_hidden_output)
    output_test = sigmoid(output_layer_input_test)

    nr_correct = np.sum(
    np.argmax(output_test, axis=1) == np.argmax(y_test10, axis=1))
    accuracy = (nr_correct / X_test10.shape[0]) * 100
    print("learning rate: ", lr/10, "accuracy: ", accuracy)
    result_lr.append(accuracy)


result_ep = []
for epoch in range(100, 1100, 100):

    weights_input_hidden, weights_hidden_output = train(
        X_train, y_train, 20, 0.1, epoch)

    # تست شبکه
    hidden_layer_input_test = np.dot(X_test10, weights_input_hidden)
    hidden_layer_output_test = sigmoid(hidden_layer_input_test)
    output_layer_input_test = np.dot(
        hidden_layer_output_test, weights_hidden_output)
    output_test = sigmoid(output_layer_input_test)

    nr_correct = np.sum(
        np.argmax(output_test, axis=1) == np.argmax(y_test10, axis=1))
    accuracy = (nr_correct / X_test10.shape[0]) * 100
    print("epoch: ", epoch, "accuracy: ", accuracy)
    result_ep.append(accuracy)

result_hd = []
for hidden_layer in range(10, 35, 5):

    weights_input_hidden, weights_hidden_output = train(
        X_train, y_train, hidden_layer, 0.1, 900)
    # تست شبکه
    hidden_layer_input_test = np.dot(X_test10, weights_input_hidden)
    hidden_layer_output_test = sigmoid(hidden_layer_input_test)
    output_layer_input_test = np.dot(
        hidden_layer_output_test, weights_hidden_output)
    output_test = sigmoid(output_layer_input_test)

    nr_correct = np.sum(
        np.argmax(output_test, axis=1) == np.argmax(y_test10, axis=1))
    accuracy = (nr_correct / X_test10.shape[0]) * 100
    print("hidden layer: ", hidden_layer, "accuracy: ", accuracy)
    result_hd.append(accuracy)



   
 # نمودار دقت شبکه بر حسب نرخ یادگیری
plt.plot(np.arange(0.1, 1.1, 0.1), result_lr, marker='o')
plt.xlabel('Learning rate')
plt.ylabel('Accuracy')
plt.title('Accuracy vs Learning rate')
plt.show()

# نمودار دقت شبکه بر حسب تعداد دوره ها
plt.plot(np.arange(100, 1100, 100), result_ep, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Accuracy vs Epochs')
plt.show()

# نمودار دقت شبکه بر حسب تعداد نورون های لایه پنهان
plt.plot(np.arange(10, 35, 5), result_hd, marker='o')
plt.xlabel('Hidden neurons')
plt.ylabel('Accuracy')
plt.title('Accuracy vs Hidden neurons')
plt.show()




