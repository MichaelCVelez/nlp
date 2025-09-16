# Assignment 2 Fall 2025 
# Note: ChatGPT used to understand concepts and as a coding assistant 
# - Michael Velez
# optimization.py

import argparse
import numpy as np
import scipy.special
import matplotlib.pyplot as plt

###
# ANSWER TO PART 1B
OPTIMAL_STEP_SIZE = 0.1
###

def _parse_args():
    """
    Command-line arguments to the system.
    :return: the parsed args bundle
    """
    parser = argparse.ArgumentParser(description='optimization.py')
    parser.add_argument('--func', type=str, default='QUAD', help='function to optimize (QUAD or NN)')
    parser.add_argument('--lr', type=float, default=1., help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=0., help='weight decay')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs')
    args = parser.parse_args()
    return args


def quadratic(x1, x2):
    """
    Quadratic function of two variables
    :param x1: first coordinate
    :param x2: second coordinate
    :return:
    """
    # y(x1, x2) = (x1 - 1)^2 + 8 * (x2 - 1)^2
    return (x1 - 1) ** 2 + 8 * (x2 - 1) ** 2


def quadratic_grad(x1, x2):
    """
    Should return a numpy array containing the gradient of the quadratic function defined above evaluated at the point
    :param x1: first coordinate
    :param x2: second coordinate
    :return: a one-dimensional numpy array containing two elements representing the gradient
    """
    # Convert to floats
    x1 = float(x1)
    x2 = float(x2)
    
    # Calculate partial derivatives using the chain and power rules
    # dy/dx1 = 2 * (x1 - 1)
    grad_x1 = 2.0 * (x1 - 1.0)
    # dy/dx2 = (8 *2) * (x2 - 1)
    grad_x2 = 16.0 * (x2 - 1.0)

    return np.array([grad_x1, grad_x2], dtype=float)


def sgd_test_quadratic(args):
    """
    Calls quadratic function inside an SGD loop and plots a visualization of the learning process.
    """
    xlist = np.linspace(-3.0, 3.0, 100)
    ylist = np.linspace(-3.0, 3.0, 100)
    X, Y = np.meshgrid(xlist, ylist)
    Z = quadratic(X, Y)
    plt.figure()

    # Track the points visited here
    points_history = []
    # Initialize at origin (0,0)
    curr_point = np.array([0., 0.])
    for epoch in range(0, args.epochs):
        grad = quadratic_grad(curr_point[0], curr_point[1])
        # Gradient must be 2D vector
        if len(grad) != 2:
            raise Exception("Gradient must be a two-dimensional array (vector containing [df/dx1, df/dx2])")
        # Gradient descent update 
        next_point = curr_point - args.lr * grad
        # Record the visited point before stepping
        points_history.append(curr_point)
        #print("Point after epoch %i: %s" % (epoch, repr(next_point)))
        curr_point = next_point
    points_history.append(curr_point)
    cp = plt.contourf(X, Y, Z)
    plt.colorbar(cp)
    plt.plot([p[0] for p in points_history], [p[1] for p in points_history], color='k', linestyle='-', linewidth=1, marker=".")
    plt.title('SGD on quadratic')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()
    #exit()

# ---------------- Helper Functions --------------------------------------------
def test_learning_rates(lrs, epochs=100, tolerance=0.1):
    """
    Helper function for testing for convergence and evaluating different learning rates 
    :param candidate_lrs: list of learning rates
    :param epochs: maximum number of iterations
    :param tolerance: distance threshold to consider convergence
    :return: dict mapping of results
    """
    optimum = np.array([1., 1.])
    results = {}

    for lr in lrs:
        # start at origin
        curr_point = np.array([0., 0.])
        steps = None
        for i in range(epochs):
            grad = quadratic_grad(curr_point[0], curr_point[1])
            # update current point
            curr_point = curr_point - lr * grad
            distance = np.linalg.norm(curr_point - optimum)
            # check for convergence
            if distance < tolerance:
                steps = i + 1
                break
        # map learning rate to number of steps or inf if no convergence 
        results[lr] = steps if steps is not None else float("inf")

    return results

def find_best_step_size():
    """
    Helper function for evaluating learning rate results and printing step size with fastest convergence.
    """    
    candidate_lrs = [0.01, 0.05, 0.1, 0.2, 0.25, 0.5]
    results = test_learning_rates(candidate_lrs)
    print("Finding best step size...")
    for lr, steps in results.items():
        status = steps if steps != float("inf") else "Learning rate did not converge."
        print(f"lr={lr:.4f} -> steps={status}")

    # Print best learning rate with fewest steps while excluding those that did not converge
    best_lr = min((lr for lr, steps in results.items() if steps != float("inf")),
                  key=lambda lr: results[lr])
    print(f"\nBest learning rate found: {best_lr:.6f} (Converged in {results[best_lr]} steps!)")
# -------------------------------------------------------------------------------

if __name__ == '__main__':
    args = _parse_args()
    find_best_step_size()
    # Hardcoded optimal step size based on results above
    args.lr = OPTIMAL_STEP_SIZE
    sgd_test_quadratic(args)
