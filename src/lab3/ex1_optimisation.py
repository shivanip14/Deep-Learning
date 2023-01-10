import tensorflow as tf
from matplotlib import pyplot as plt
from tqdm import trange

base_path = './lab3data/'

# Provided code, modified to work with tensorflow2
def do_stuff(optimizer):
    # training data
    x_train = [1, 2, 3, 4]
    y_train = [0, -1, -2, -3]

    # Model parameters
    W = tf.Variable([.3], dtype=tf.float32)
    b = tf.Variable([-.3], dtype=tf.float32)
    # Model input and output
    x = tf.compat.v1.placeholder(tf.float32)
    linear_model = W * x + b
    y = tf.compat.v1.placeholder(tf.float32)

    # loss
    loss = tf.reduce_sum(tf.square(linear_model - y))  # sum of the squares
    # optimizer
    train = optimizer.minimize(loss)

    # training loop
    init = tf.compat.v1.global_variables_initializer()
    sess = tf.compat.v1.Session()
    sess.run(init)  # reset values to wrong

    curr_W, curr_b, curr_loss = sess.run([W, b, loss], {x: x_train, y: y_train})
    #print("W: %s b: %s loss: %s" % (curr_W, curr_b, curr_loss))

    losses = []
    for i in trange(1000):
        sess.run(train, {x: x_train, y: y_train})
        curr_W, curr_b, curr_loss = sess.run([W, b, loss], {x: x_train, y: y_train})
        # print("W: %s b: %s loss: %s" % (curr_W, curr_b, curr_loss))
        losses.append(curr_loss)

    return losses

# Takes in a dictionary, with name being the key, and values being a list
def plot_all(vals, title):
    plt.title(title)
    plt.xlabel("Iteration")
    plt.ylabel("Loss Function Value")
    for name in vals:
        points = vals[name]
        plt.plot(points, label=name)
    plt.legend()
    plt.savefig(base_path + '/ex1/loss_graphs_' + title + '.pdf')
    plt.close()

def plot_all_01(vals):
    plt.title("Different optimizers at 0.01 learning rate")
    plt.xlabel("Iteration")
    plt.ylabel("Loss Function Value")
    for name in vals:
        points = vals[name]
        plt.plot(points, label=name)
    plt.legend()
    plt.savefig(base_path + '/ex1/loss_graphs_01.pdf')
    plt.close()

# looping through different optimizers and learning rates here.
def main():
    losses_01 = {}
    # change lrs, and optimizers as per requirement
    lrs = [1.0, 0.1, 0.01, 0.001, 0.0001]
    optimizers = {"GradientDescent": tf.compat.v1.train.GradientDescentOptimizer,
                  "AdaGrad": tf.compat.v1.train.AdagradOptimizer,
                  "Adam": tf.compat.v1.train.AdamOptimizer,
                  "RMSprop": tf.compat.v1.train.RMSPropOptimizer,
                  "Momentum": tf.compat.v1.train.MomentumOptimizer,
                  "AdaDelta": tf.compat.v1.train.AdadeltaOptimizer,
                  "FTRL": tf.compat.v1.train.FtrlOptimizer,
                  "ProximalAdaGrad": tf.compat.v1.train.ProximalAdagradOptimizer,
                  "ProximalGradientDescent": tf.compat.v1.train.ProximalGradientDescentOptimizer}

    for optim_name in optimizers:
        all_losses = {}
        for lr in lrs:
            title = optim_name + " with LR " + str(lr)
            if optim_name == "Momentum":
                optimizer = optimizers[optim_name](learning_rate=lr, momentum=0.9)
            else:
                optimizer = optimizers[optim_name](lr)
            # run optimizer as it is
            print("Running " + title)
            losses = do_stuff(optimizer)
            all_losses[str(lr)] = losses
            if lr == 0.01:
                losses_01[optim_name] = losses
        plot_all(all_losses, title=optim_name)
    plot_all_01(losses_01)


if __name__ == "__main__":
    # This change makes the provided code compatible with tf2
    tf.compat.v1.disable_eager_execution()
    print("Using tensorFlow version " + str(tf.__version__))
    main()
