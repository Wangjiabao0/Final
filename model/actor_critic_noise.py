import torch
import torch.nn as nn
import torch.nn.functional as F

class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, sigma_init=0.017):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        # learnable parameters
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))

        # Noise parameters
        self.register_buffer("weight_epsilon", torch.empty(out_features, in_features))
        self.register_buffer("bias_epsilon", torch.empty(out_features))

        self.sigma_init = sigma_init
        self.reset_parameters()

    def reset_parameters(self):
        # initialize µ 
        nn.init.uniform_(self.weight_mu, -1 / self.in_features**0.5, 1 / self.in_features**0.5)
        nn.init.uniform_(self.bias_mu, -1 / self.in_features**0.5, 1 / self.in_features**0.5)
        # initialize σ
        nn.init.constant_(self.weight_sigma, self.sigma_init)
        nn.init.constant_(self.bias_sigma, self.sigma_init)

    def forward(self, x):
        if self.training:
            self.weight_epsilon.normal_()
            self.bias_epsilon.normal_()
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return F.linear(x, weight, bias)
    
class ActorCriticNoise(nn.Module):
    def __init__(self, config, num_of_actions):
        super().__init__()
        num_of_inputs = config.stack_size
        sigma_init = config.sigma_init
        self.conv1 = nn.Conv2d(num_of_inputs, 16, 8, stride=4)
        self.conv2 = nn.Conv2d(16, 32, 3, stride=2)
        self.linear1 = NoisyLinear(32 * 9 * 9, 256, sigma_init=sigma_init) 
        self.policy = NoisyLinear(256, num_of_actions, sigma_init=sigma_init)
        self.value = NoisyLinear(256, 1, sigma_init=sigma_init) 

    def forward(self, x):
        conv1_out = F.relu(self.conv1(x))
        conv2_out = F.relu(self.conv2(conv1_out))

        flattened = torch.flatten(conv2_out, start_dim=1)
        linear1_out = F.relu(self.linear1(flattened))

        policy_output = self.policy(linear1_out)
        value_output = self.value(linear1_out)

        probs = F.softmax(policy_output, dim=1)
        log_probs = F.log_softmax(policy_output, dim=1)
        return probs, log_probs, value_output, None