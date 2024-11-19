# 24 Fall
# CS 5330 - Project 5
# Sihe Chen
# 002085773
# chen.sihe1@northeastern.edu
# generate different combinations of parameters for experiment

all_params = {
    "num_epochs": (5, 10, 15),
    "batch_train_size": (32, 64, 128),
    "num_hidden_nodes": (50, 100, 150),
    "dropout_rate": (0.3, 0.5, 0.7),
    "activation_function": ("tanh", "relu", "sigmoid"),
    "loss_function": ("cross_entropy", "nll", "mse"),
    "learning_rate": (0.001, 0.01, 0.1),
    "momentum": (0.3, 0.5, 0.7),
}

idx = 0
all_combs = []
for param, vals in all_params.items():
    for i in range(len(vals)):
        arguments = f"--{param} {vals[i]} "
        for other_param, other_vals in all_params.items():
            if other_param != param:
                arguments += f"--{other_param} {other_vals[1]} "
        arguments += f"--exp_id {idx}"
        idx += 1
        all_combs.append(arguments)
        print(arguments)


conv_layer_param_names = ("num_conv_layers", "num_conv_filters", "conv_filter_sizes")

conv_layer_params = [
    (2, "10,20", "3,3"),
    (2, "15,30", "3,3"),
    (2, "20,40", "3,3"),
    (2, "10,20", "4,4"),
    (2, "10,20", "5,5"),
    (3, "10,20,20", "3,3,3"),
    (4, "10,20,20,20", "3,3,3,3")
]

for i in range(len(conv_layer_params)):
    arguments = ""
    for j in range(3):
        arguments += f"--{conv_layer_param_names[j]} {conv_layer_params[i][j]} "
    arguments += f"--exp_id {idx}"
    idx += 1
    print(arguments)

