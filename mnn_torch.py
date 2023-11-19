import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_squared_error


class MemoryNeuralNetwork(nn.Module):
    def __init__(self, number_of_input_neurons=2, number_of_hidden_neurons=6, number_of_output_neurons=2, neeta=4e-5, neeta_dash=4e-5, lipschitz_norm=1.0, spectral_norm=False, seed_value=16981):
        super(MemoryNeuralNetwork, self).__init__()

        torch.manual_seed(seed_value)

        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")

        self.number_of_input_neurons = number_of_input_neurons
        self.number_of_hidden_neurons = number_of_hidden_neurons
        self.number_of_output_neurons = number_of_output_neurons
        self.squared_error = 0.0

        self.spectral_norm = spectral_norm
        self.lipschitz = lipschitz_norm

        self.neeta = neeta
        self.neeta_dash = neeta_dash

        # Initialize memory coefficients randomly
        self.alpha_input_layer = nn.Parameter(
            torch.rand(self.number_of_input_neurons, device=self.device))
        self.alpha_hidden_layer = nn.Parameter(
            torch.rand(self.number_of_hidden_neurons, device=self.device))
        self.alpha_last_layer = nn.Parameter(
            torch.rand(self.number_of_output_neurons, device=self.device))

        # Initialize weights of the network randomly
        self.beta = nn.Parameter(torch.rand(self.number_of_output_neurons, device=self.device))
        self.weights_input_to_hidden_nn = nn.Parameter(torch.rand(
            self.number_of_input_neurons, self.number_of_hidden_neurons, device=self.device))
        self.weights_hidden_to_output_nn = nn.Parameter(torch.rand(
            self.number_of_hidden_neurons, self.number_of_output_neurons, device=self.device))
        self.weights_input_to_hidden_mn = nn.Parameter(torch.rand(
            self.number_of_input_neurons, self.number_of_hidden_neurons, device=self.device))
        self.weights_hidden_to_output_mn = nn.Parameter(torch.rand(
            self.number_of_hidden_neurons, self.number_of_output_neurons, device=self.device))

        # Initialize past values as zeros
        self.prev_output_of_input_layer_nn = torch.zeros(
            self.number_of_input_neurons, device=self.device)
        self.prev_output_of_input_layer_mn = torch.zeros(
            self.number_of_input_neurons, device=self.device)
        self.prev_output_of_hidden_layer_nn = torch.zeros(
            self.number_of_hidden_neurons, device=self.device)
        self.prev_output_of_hidden_layer_mn = torch.zeros(
            self.number_of_hidden_neurons, device=self.device)
        self.prev_output_of_nn = torch.zeros(
            self.number_of_output_neurons, device=self.device)
        self.prev_output_of_mn = torch.zeros(
            self.number_of_output_neurons, device=self.device)

        self.to(self.device)

    def feedforward(self, input_array):
        with torch.no_grad():
            # if self.device.type == 'cuda':
            #     # print(torch.cuda.get_device_name(0))
            #     print('Memory Usage:\n')
            #     print('Allocated:', round(
            #         torch.cuda.memory_allocated(0)/1024**3, 1), 'GB\n')
            #     print('Cached:   ', round(
            #         torch.cuda.memory_reserved(0)/1024**3, 1), 'GB\n')

            self.input_nn = torch.tensor(
                input_array, dtype=torch.float32, device=self.device)
            self.output_of_input_layer_nn = self.input_nn
            self.output_of_input_layer_mn = (self.alpha_input_layer * self.prev_output_of_input_layer_nn) + (
                (1.0 - self.alpha_input_layer) * self.prev_output_of_input_layer_mn)

            self.input_to_hidden_layer_nn = torch.matmul(self.weights_input_to_hidden_nn.t(
            ), self.output_of_input_layer_nn) + torch.matmul(self.weights_input_to_hidden_mn.t(), self.output_of_input_layer_mn)
            self.output_of_hidden_layer_nn = self.activation_function(
                self.input_to_hidden_layer_nn)
            self.output_of_hidden_layer_mn = (self.alpha_hidden_layer * self.prev_output_of_hidden_layer_nn) + (
                (1.0 - self.alpha_hidden_layer) * self.prev_output_of_hidden_layer_mn)

            self.output_of_last_layer_mn = (self.alpha_last_layer * self.prev_output_of_nn) + (
                (1.0 - self.alpha_last_layer) * self.prev_output_of_mn)
            self.input_to_last_layer_nn = torch.matmul(self.weights_hidden_to_output_nn.t(), self.output_of_hidden_layer_nn) + torch.matmul(
                self.weights_hidden_to_output_mn.t(), self.output_of_hidden_layer_mn) + (self.beta * self.output_of_last_layer_mn)
            self.output_nn = self.output_layer_activation_function(
                self.input_to_last_layer_nn).to(device=self.device)

            self.prev_output_of_input_layer_nn = self.output_of_input_layer_nn.clone()
            self.prev_output_of_input_layer_mn = self.output_of_input_layer_mn.clone()
            self.prev_output_of_hidden_layer_nn = self.output_of_hidden_layer_nn.clone()
            self.prev_output_of_hidden_layer_mn = self.output_of_hidden_layer_mn.clone()
            self.prev_output_of_nn = self.output_nn.clone()
            self.prev_output_of_mn = self.output_of_last_layer_mn.clone()

            return self.output_nn

    def backprop(self, y_des):
        with torch.no_grad():
            # if self.device.type == 'cuda':
            #     # print(torch.cuda.get_device_name(0))
            #     print('Memory Usage:\n')
            #     print('Allocated:', round(
            #         torch.cuda.memory_allocated(0)/1024**3, 3), 'GB\n')
            #     print('Cached:   ', round(
            #         torch.cuda.memory_reserved(0)/1024**3, 3), 'GB\n')
                
            self.y_des = torch.tensor(
                y_des, dtype=torch.float32, device=self.device,requires_grad=False)
            
            self.rmse_x=mean_squared_error([self.y_des.cpu().detach().numpy()[0]],[self.output_nn.cpu().detach().numpy()[0]],squared=False)
            self.rmse_y=mean_squared_error([self.y_des.cpu().detach().numpy()[1]],[self.output_nn.cpu().detach().numpy()[1]],squared=False)
            self.rmse_z=mean_squared_error([self.y_des.cpu().detach().numpy()[2]],[self.output_nn.cpu().detach().numpy()[2]],squared=False)
            self.rmse = torch.sqrt(torch.mean((self.y_des - self.output_nn)**2))

            self.error_last_layer = (self.output_nn - self.y_des) * \
                self.output_layer_activation_function_derivative(
                    self.input_to_last_layer_nn)
            self.error_hidden_layer = self.activation_function_derivative(
                self.input_to_hidden_layer_nn) * torch.matmul(self.weights_hidden_to_output_nn, self.error_last_layer)
            
            #print(self.error_last_layer.repeat(20,1).size(),self.output_of_hidden_layer_nn.repeat(3,1).t())
            
            #Update Weights of network
            self.weights_hidden_to_output_nn -= self.neeta *self.error_last_layer.repeat(self.number_of_hidden_neurons,1) *self.output_of_hidden_layer_nn.repeat(self.number_of_output_neurons,1).t()
            self.weights_input_to_hidden_nn -= self.neeta *self.error_hidden_layer.repeat(self.number_of_input_neurons,1) *self.output_of_input_layer_nn.repeat(self.number_of_hidden_neurons,1).t()
            self.weights_hidden_to_output_mn -= self.neeta *self.error_last_layer.repeat(self.number_of_hidden_neurons,1) *self.output_of_hidden_layer_mn.repeat(self.number_of_output_neurons,1).t()
            self.weights_input_to_hidden_mn -= self.neeta *self.error_hidden_layer.repeat(self.number_of_input_neurons,1) *self.output_of_input_layer_mn.repeat(self.number_of_hidden_neurons,1).t()
            self.beta -= self.neeta_dash * self.error_last_layer * self.output_of_last_layer_mn

            #pd means partial derivative
            self.pd_e_wrt_v_hidden_layer = torch.matmul(
                self.weights_hidden_to_output_mn, self.error_last_layer)
            self.pd_e_wrt_v_input_layer = torch.matmul(
                self.weights_input_to_hidden_mn, self.error_hidden_layer)
            self.pd_e_wrt_v_last_layer = self.beta * self.error_last_layer
            self.pd_v_wrt_alpha_hidden_layer = self.prev_output_of_hidden_layer_nn - \
                self.prev_output_of_hidden_layer_mn
            self.pd_v_wrt_alpha_input_layer = self.prev_output_of_input_layer_nn - \
                self.prev_output_of_input_layer_mn
            self.pd_v_wrt_alpha_last_layer = self.prev_output_of_nn - self.prev_output_of_mn

            self.alpha_hidden_layer -= self.neeta_dash * \
                self.pd_e_wrt_v_hidden_layer * self.pd_v_wrt_alpha_hidden_layer
            self.alpha_input_layer -= self.neeta_dash * \
                self.pd_e_wrt_v_input_layer * self.pd_v_wrt_alpha_input_layer
            self.alpha_last_layer -= self.neeta_dash * \
                self.pd_e_wrt_v_last_layer * self.pd_v_wrt_alpha_last_layer

            self.alpha_hidden_layer.data = torch.clamp(
                self.alpha_hidden_layer, 0.0, 1.0)
            self.alpha_input_layer.data = torch.clamp(
                self.alpha_input_layer, 0.0, 1.0)
            self.alpha_last_layer.data = torch.clamp(
                self.alpha_last_layer, 0.0, 1.0)
            self.beta.data = torch.clamp(self.beta, 0.0, 1.0)

            if self.spectral_norm:
                self.weights_input_to_hidden_mn.data = (self.weights_input_to_hidden_mn / torch.norm(
                    self.weights_input_to_hidden_mn, p=2)) * (self.lipschitz ** (1/2))
                self.weights_input_to_hidden_nn.data = (self.weights_input_to_hidden_nn / torch.norm(
                    self.weights_input_to_hidden_nn, p=2)) * (self.lipschitz ** (1/2))
                self.weights_hidden_to_output_mn.data = (self.weights_hidden_to_output_mn / torch.norm(
                    self.weights_hidden_to_output_mn, p=2)) * (self.lipschitz ** (1/2))
                self.weights_hidden_to_output_nn.data = (self.weights_hidden_to_output_nn / torch.norm(
                    self.weights_hidden_to_output_nn, p=2)) * (self.lipschitz ** (1/2))
                self.beta.data = (self.beta / torch.norm(self.beta,
                                p=2)) * (self.lipschitz ** (1/2))

    def activation_function(self, x):
        with torch.no_grad():
            g1_x = 15 * torch.tanh(x / 15)
            return g1_x

    def output_layer_activation_function(self, x):
        return x

    def activation_function_derivative(self, x):
        with torch.no_grad():
            return 1.0 - torch.square(torch.tanh(x / 15))

    def output_layer_activation_function_derivative(self, x):
        return 1.0
