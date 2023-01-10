import tensorflow as tf
import read_inputs
import numpy as np
from matplotlib import pyplot as plt

base_path = './lab3data/'

# Takes in a dictionary, with name being the key, and values being a list
def plot_all(loss_vals, acc_vals, title):
    plt.title(title)
    plt.xlabel("Iteration")
    plt.ylabel("Accuracy")
    for name in acc_vals:
        points = acc_vals[name]
        plt.plot(points, label=name + ' (' + str(points[-1]) + ')')
    plt.legend()
    plt.savefig(base_path + '/ex2/accuracy_graphs_' + title + '.pdf')
    plt.close()

    plt.title(title)
    plt.xlabel("Iteration")
    plt.ylabel("Cross-entropy Loss")
    for name in loss_vals:
        points = loss_vals[name]
        plt.plot(points, label=name + ' (' + str(points[-1]) + ')')
    plt.legend()
    plt.savefig(base_path + '/ex2/loss_graphs_' + title + '.pdf')
    plt.close()

def plot_all_01(loss_vals, acc_vals):
    plt.title("Different optimizers at 0.01 learning rate")
    plt.xlabel("Iteration")
    plt.ylabel("Accuracy")
    for name in acc_vals:
        points = acc_vals[name]
        plt.plot(points, label=name + ' (' + str(points[-1]) + ')')
    plt.legend()
    plt.savefig(base_path + '/ex2/accuracy_graphs_01.pdf')
    plt.close()

    plt.title("Different optimizers at 0.01 learning rate")
    plt.xlabel("Iteration")
    plt.ylabel("Cross-entropy Loss")
    for name in loss_vals:
        points = loss_vals[name]
        plt.plot(points, label=name + ' (' + str(points[-1]) + ')')
    plt.legend()
    plt.savefig(base_path + '/ex2/loss_graphs_01.pdf')
    plt.close()

def do_stuff(optimizing_function):
    data_input = read_inputs.load_data_mnist(base_path + 'MNIST_data/mnist.pkl.gz')
    data = data_input[0]

    real_output = np.zeros((np.shape(data[0][1])[0], 10), dtype=np.float)
    for i in range(np.shape(data[0][1])[0]):
        real_output[i][data[0][1][i]] = 1.0

    real_check = np.zeros((np.shape(data[2][1])[0], 10), dtype=np.float)
    for i in range(np.shape(data[2][1])[0]):
        real_check[i][data[2][1][i]] = 1.0

    x = tf.compat.v1.placeholder(tf.float32, [None, 784])
    W = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))
    y = tf.nn.softmax(tf.matmul(x, W) + b)
    y_ = tf.compat.v1.placeholder(tf.float32, [None, 10])

    cross_entropy = tf.reduce_mean(-tf.compat.v1.reduce_sum(y_ * tf.math.log(y), reduction_indices=[1]))
    train_step = optimizing_function.minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    sess = tf.compat.v1.InteractiveSession()
    tf.compat.v1.global_variables_initializer().run()
    accuracies = []
    losses = []
    # training phase
    print("Training")
    for i in range(500):
        batch_xs = data[0][0][100 * i:100 * i + 100]
        batch_ys = real_output[100 * i: 100 * i + 100]
        curr_train_step, curr_cross_entropy = sess.run([train_step, cross_entropy], feed_dict={x: batch_xs, y_: batch_ys})
        losses.append(curr_cross_entropy)
        accuracy_val = sess.run(accuracy, feed_dict={x: data[2][0], y_: real_check})
        accuracies.append(accuracy_val)

    return losses, accuracies


def main():
    lrs = [1.0, 0.1, 0.01, 0.001, 0.0001]
    optimizers = {"GradientDescent": tf.compat.v1.train.GradientDescentOptimizer,
                  "AdaGrad": tf.compat.v1.train.AdagradOptimizer,
                  "Adam": tf.compat.v1.train.AdamOptimizer,
                  "RMSprop": tf.compat.v1.train.RMSPropOptimizer,
                  "Momentum": tf.compat.v1.train.MomentumOptimizer,
                  "AdaDelta": tf.compat.v1.train.AdadeltaOptimizer}
                  #"FTRL": tf.compat.v1.train.FtrlOptimizer,
                  #"ProximalAdaGrad": tf.compat.v1.train.ProximalAdagradOptimizer,
                  #"ProximalGradientDescent": tf.compat.v1.train.ProximalGradientDescentOptimizer}
    accuracy_01 = {}
    loss_01 = {}

    for optim_name in optimizers:
        all_acc = {}
        all_loss = {}
        for lr in lrs:
            title = optim_name + " LR " + str(lr)
            if optim_name == "Momentum":
                optimizer = optimizers[optim_name](learning_rate=lr, momentum=0.95)
            elif optim_name == "AdaGrad":
                optimizer = optimizers[optim_name](learning_rate=lr, initial_accumulator_value=0.1)
            elif optim_name == "Adam":
                optimizer = optimizers[optim_name](learning_rate=lr, beta1=0.9, beta2=0.999, epsilon=1e-7)
            elif optim_name == "RMSprop":
                optimizer = optimizers[optim_name](learning_rate=lr, decay=0.9, momentum=0.1, epsilon=1e-7, centered=False)
            elif optim_name == "AdaDelta":
                optimizer = optimizers[optim_name](learning_rate=lr, rho=0.95, epsilon=1e-7)
            else:
                optimizer = optimizers[optim_name](lr)
            print('Running ' + title)
            loss, accuracy = do_stuff(optimizer)
            all_loss[str(lr)] = loss
            all_acc[str(lr)] = accuracy
            if lr == 0.01:
                accuracy_01[optim_name] = accuracy
                loss_01[optim_name] = loss
        plot_all(all_loss, all_acc, title=optim_name)
    plot_all_01(loss_01, accuracy_01)


if __name__ == "__main__":
    # This change makes the provided code compatible with tf2
    tf.compat.v1.disable_eager_execution()
    print("Using tensorFlow version " + str(tf.__version__))
    main()
