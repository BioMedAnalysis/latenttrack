import torch
import typing
import numpy as np
from model.model import Encoder, Decoder, Seq2Seq
from config import MODEL_CONF
from utils import embedding


class ModelInference(object):
    def __init__(self, weights_file: str, device: str = "cpu") -> None:
        self.weights_file = weights_file
        self.device = device

        self.encoder = Encoder(
            MODEL_CONF["encoder"]["input_dim"],
            MODEL_CONF["encoder"]["hidden_dim"],
            MODEL_CONF["encoder"]["n_layer"],
            MODEL_CONF["encoder"]["dropout"],
        )

        self.decoder = Decoder(
            MODEL_CONF["decoder"]["input_dim"],
            MODEL_CONF["decoder"]["output_dim"],
            MODEL_CONF["decoder"]["hidden_dim"],
            MODEL_CONF["decoder"]["n_layer"],
            MODEL_CONF["decoder"]["dropout"],
        )

        self.net = Seq2Seq(self.encoder, self.decoder, self.device)

        # load weight
        self.net.load_state_dict(
            torch.load(self.weights_file, map_location=self.device)
        )
        print("weights loaded.")

    def tck2emb(self, streamlines: typing.List) -> np.array:
        return embedding(self.net, streamlines)
