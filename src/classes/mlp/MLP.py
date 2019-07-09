import sys, math
import random
import numpy as np


# from classes.Layer import Layer

class MLP:
    input_layer_size = 3
    hidden_layer_size = 3
    output_layer_size = 1


    # CONSTRUCTOR:

    def __init__(self, learning_examples_array):
        self.learning_examples_array = learning_examples_array
        self.init_weights()

    def init_weights(self):

        self.weights = [None, None]

        self.weights[0] = np.array([
            # Input layer to hidden layer

            # i1 connections
            [random.random(), random.random(), random.random()],

            # i2 connections
            [random.random(), random.random(), random.random()],

            # i3 connections
            [random.random(), random.random(), random.random()],
        ])

        self.weights[1] = np.array([
            # Hidden layer to output layer

            # h1 connections
            [random.random()],

            # h2 connections
            [random.random()],

            # h3 connections
            [random.random()]

        ])

        # FORWARD FEED - NODE VALUES:
        # important: use 0.0 instead of 0 (otherwise array dtype will be int)
        self.node_values = [None, None, None]
        self.node_values[0] = np.array([0.0, 0.0, 0.0])
        self.node_values[1] = np.array([0.0, 0.0, 0.0])
        self.node_values[2] = np.array([0.0])

        self.weights_gradients = [None, None]
        self.weights_gradients[0] = np.array([
            # Input layer to hidden layer

            # i1 connections
            [0.0, 0.0, 0.0, 0.0],

            # i2 connections
            [0.0, 0.0, 0.0, 0.0],

            # i3 connections
            [0.0, 0.0, 0.0, 0.0],
        ])

        self.weights_gradients[1] = np.array([
            # Hidden layer to output layer

            # h1 connections
            [0.0],

            # h2 connections
            [0.0],

            # h3 connections
            [0.0],

            # h4 connections
            [0.0]

        ])

    # FORWARD FEED:

    def ff_apply_inputs(self, image_info_row):
        # print('Adding inputs to the beginning of the matrix')
        # print('image row:')

        self.node_values[0][0] = image_info_row[1]
        self.node_values[0][1] = image_info_row[2]
        self.node_values[0][2] = image_info_row[3]

        self.node_values[0] = np.array([image_info_row[1], image_info_row[2], image_info_row[3]])

    def ff_compute_hidden_layer(self):
        for i in range(0, self.hidden_layer_size):
            column_vector = self.weights[0][:, [i]]
            column_vector = column_vector.transpose()
            # W * x (where x is inputs vector)

            new_node_value = np.matmul(column_vector, np.array(self.node_values[0]))[0]

            new_node_value = self.activation_function(new_node_value)

            self.node_values[1][i] = new_node_value

        # print('Computing hidden layer node values...')

    def ff_compute_output_layer(self):
        for i in range(0, self.output_layer_size):
            column_vector = self.weights[1][:, [i]]
            column_vector = column_vector.transpose()
            # W * x (where x is hidden layer node values vector)
            new_node_value = np.matmul(column_vector, self.node_values[1])[0]

            # new_node_value = self.activation_function(new_node_value)

            # update output node value:
            self.node_values[2][i] = new_node_value
        # print('Computing output layer node values...')

    # basic difference between target ouput value and the obtained (computed) output value:
    def compute_delta(self, target_value, computed_value):
        return (target_value - computed_value)

    # BACKPROPAGATION:
    def bp_compute_output_layer_gradients(self, target_value, output_value_index):
        #
        # print('rvo=' + str( self.node_values[2][output_value_index]))
        # print('targeto=' + str( target_value))

        delta = self.compute_delta(target_value, self.node_values[2][output_value_index])
        # print('deltao=' + str(delta))

        for i in range(0, self.hidden_layer_size):
            gradient_value = - 2 * delta * self.node_values[1][i]
            # print('gvo=' + str(gradient_value))
            self.weights_gradients[1][i][output_value_index] = gradient_value

        # print('Computing output layer gradients')

    def bp_compute_hidden_layer_gradients(self, target_value, output_value_index):
        delta = self.compute_delta(target_value, self.node_values[2][output_value_index])

        for i in range(0, self.input_layer_size):
            for j in range(0, self.hidden_layer_size):
                column_vector = self.weights[0][:, [j]]
                column_vector = column_vector.transpose()
                input_vector = np.array(self.node_values[0])
                pre_activation_node_value = np.matmul(column_vector, input_vector)[0]

                current_input_val = self.node_values[0][i]
                gradient_value = - 2 * delta * self.node_values[1][j]

                # print('part1 = ' + str(gradient_value))

                gradient_value = gradient_value * self.activation_function_derivative(
                    pre_activation_node_value) * current_input_val

                # print('after part 2 = ' + str(gradient_value))

                # print('gvh=' + str(gradient_value))

                self.weights_gradients[0][i][j] = gradient_value

        # print('Computing hidden layer gradients')

    def bp_update_weights(self):
        learning_rate = 0.2

        for layer in range(0, 2):
            for i in range(self.weights[layer].shape[0]):
                for j in range(self.weights[layer].shape[1]):
                    gradient_value = self.weights_gradients[layer][i][j]
                    #
                    # print('layer=' + str(layer) + ' and i=' + str(i) + ', j=' + str(j) + ' and gv=' + str(
                    #     gradient_value))

                    if (gradient_value > 0):
                        self.weights[layer][i][j] += -learning_rate * abs(gradient_value)
                        # self.weights[layer][i][j] += -learning_rate
                    elif (gradient_value < 0):
                        self.weights[layer][i][j] += learning_rate * abs(gradient_value)
                        # self.weights[layer][i][j] += learning_rate

        # print('Updating weights..')

    # CORE API:

    def train_network(self):
        print('Training network...')
        print('learning examples array:')
        for i in range(0, self.learning_examples_array.shape[0]):
            # for i in range(0, 1):
            target_value = self.learning_examples_array[i][4]
            # target_value = target_value * 3.0

            # Forward pass:

            self.ff_apply_inputs(self.learning_examples_array[i])
            self.ff_compute_hidden_layer()
            self.ff_compute_output_layer()

            for j in range(0, self.output_layer_size):
                self.bp_compute_output_layer_gradients(target_value, j)
                self.bp_compute_hidden_layer_gradients(target_value, j)

            self.bp_update_weights()

            # print('FULL NODE VALUE LIST:')
            # print(self.node_values)
            #
            # print('FULL GRADIENTS VALUE LIST:')
            # print(self.weights_gradients)
            #
            # print('WEIGHTS LIST:')
            # print(self.weights)

            # print(self.weights[0][2][2])

        # print('FINAL WEIGHTS:')
        # print(self.weights)
        #
        # print(self.weights[0][2][2])

    def predict(self, image_object):
        # print('Predicting...')

        self.ff_apply_inputs(image_object)
        self.ff_compute_hidden_layer()
        self.ff_compute_output_layer()

        output_value = self.node_values[2][0]
        # print('OUTPUT VALUE:')
        # print(output_value)

        # print('PREDICTING ON WEIGHTS:')
        # print(self.weights)

        return output_value


    def calculate_total_error_on_dataset(self, dataset):

        total_delta = 0.0
        total_loss = 0.0

        for i in range(0, self.learning_examples_array.shape[0]):

            row  =self.learning_examples_array[i]
            target_value = row[4]

            predicted_value = self.predict(row)
            total_delta += self.compute_delta(target_value, predicted_value)
            total_loss += self.loss_function(target_value,predicted_value)

        return (total_delta, total_loss)



    # MATH FUNCTIONS:

    # squared(target output - computed output)
    def loss_function(self, target_value, computed_value):
        return math.pow(target_value - computed_value, 2)

    # logistic function
    def activation_function(self, x):
        return 1.0 / (1.0 + math.exp(-x))

    # derivative of logistic function g(z)' = g(z) * (1 - g(z))
    def activation_function_derivative(self, x):
        return self.activation_function(x) * (1 - self.activation_function(x))


