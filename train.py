import argparse
import torch
from random import shuffle
from torch import nn
from torch import optim
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from model import Encoder, Decoder, Seq2Seq
from utils.dataset import FileScanner, tck2dataset
from config import MODEL_CONF

# training parameters
training_ratio = 0.8
validation_ratio = 0.1
test_ratio = 0.1

batch_size = 128
epoch = 10
save_every = 100
export_every = 300

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tbwriter = SummaryWriter("./tfboard")


def normalize(_in):
    return (_in + np.array([180, 216, 0])) / np.array(
        [180, 216, 180]
    )  # only for simulated data


def train(model, x, y, optimizer, criterion, clip):

    model.train()

    epoch_loss = 0

    src = x.to(device)
    trg = y.to(device)

    optimizer.zero_grad()

    output = model(src, trg)

    loss = criterion(output, trg)

    loss.backward()

    torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

    optimizer.step()

    epoch_loss += loss.item()

    return epoch_loss / batch_size


def main(tck_dir):
    # define model
    encoder = Encoder(
        MODEL_CONF["encoder"]["input_dim"],
        MODEL_CONF["encoder"]["hidden_dim"],
        MODEL_CONF["encoder"]["n_layer"],
        MODEL_CONF["encoder"]["dropout"],
    )

    decoder = Decoder(
        MODEL_CONF["decoder"]["input_dim"],
        MODEL_CONF["decoder"]["output_dim"],
        MODEL_CONF["decoder"]["hidden_dim"],
        MODEL_CONF["decoder"]["n_layer"],
        MODEL_CONF["decoder"]["dropout"],
    )
    model = Seq2Seq(encoder, decoder, device)
    model = model.to(device)

    # prepare dataset
    tck_files = FileScanner.scan(tck_dir)
    combined_streamlines = tck2dataset(tck_files, normalize)
    shuffle(combined_streamlines)  # shuffle the data
    print(f"total streamlines: {len(combined_streamlines)}")
    print(f"steps per epoch: {len(combined_streamlines) // batch_size}")

    # training
    # init the weights
    def init_weights(m):
        for name, param in m.named_parameters():
            nn.init.uniform_(param.data, -0.08, 0.08)

    model.apply(init_weights)

    # count parameters
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("The model has {} trainable parameters".format(count_parameters(model)))

    optimizer = optim.Adam(model.parameters())
    criterion = nn.MSELoss()
    CLIP = 1.0

    # main train process
    losses = []

    global_id = 0
    for _p in range(1, 10):
        shuffle(combined_streamlines)  # shuffle the streamlines
        total_length = len(combined_streamlines)
        # each epoch
        for i in range(0, total_length, batch_size):
            whole = combined_streamlines[i : i + batch_size]

            x_list = []
            y_list = []

            for each in whole:
                length = each.size(0)
                middle = length // 2
                each.shape[0]
                x_list.append(torch.cat((each[:middle], torch.zeros(1, 3))))
                y_list.append(torch.cat((torch.zeros(1, 3), each[middle:])))

            x = torch.nn.utils.rnn.pad_sequence(x_list, batch_first=True)
            x = x.permute((1, 0, 2))
            y = torch.nn.utils.rnn.pad_sequence(y_list, batch_first=True)
            y = y.permute((1, 0, 2))

            if y.shape[1] != 128:
                break

            # x = torch.cat((y, torch.zeros(1, batch_size, 3)))

            _loss = train(model, x, y, optimizer, criterion, CLIP)

            losses.append(_loss)

            # print loss
            if (i / batch_size) % save_every == 0:
                print("iteration {}: {}".format(i, _loss))
                tbwriter.add_scalar("training loss", _loss, global_id)

            # save the weights
            if (i / batch_size) % export_every == 0:
                torch.save(
                    model.state_dict(),
                    "./model-weights-half-simulated-{}-{}.pt".format(_p, i),
                )
                print("saved a weight snapshoot iteration {}-{}".format(_p, i))

            global_id +=1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tck_file_dir",
        help="The path to directory containing the tck file",
        type=str,
        required=True,
    )
    args = parser.parse_args()

    main(args.tck_file_dir)
