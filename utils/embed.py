import torch


def embedding(model, streamlines, half=True):
    
    if half:
        
        x_list = []
        
        for each in streamlines:
            length = each.size(0)
            middle = length // 2
            each.shape[0]
            x_list.append(torch.cat((each[:middle], torch.zeros(1, 3))))
        
        x = torch.nn.utils.rnn.pad_sequence(x_list, batch_first=True)
        x = x.permute((1,0,2))
        
        embed = model.encoder(x)
    else:
        padded_seq_batch = torch.nn.utils.rnn.pad_sequence(streamlines, batch_first=True)
        padded_seq_batch = padded_seq_batch.permute((1,0,2))

        embed = model.encoder(padded_seq_batch)
    
    #return torch.cat((embed[0], embed[1]), dim=2)
    return embed[0] # hidden state only
    #return embed[1] # cell state only

