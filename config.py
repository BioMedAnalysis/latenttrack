MODEL_CONF = {
    "encoder": {"input_dim": 3, "hidden_dim": 128, "n_layer": 1, "dropout": 0},
    "decoder": {
        "input_dim": 3,
        "output_dim": 3,
        "hidden_dim": 128,
        "n_layer": 1,  # default 1
        "dropout": 0,
    },
}
