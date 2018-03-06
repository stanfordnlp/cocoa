import onmt
from onmt.Utils import use_gpu

def make_loss(opt, mappings, model):
    vocab = mappings["vocab"]
    padding_idx = vocab.word_to_ind["<pad>"]
    loss = onmt.Loss.NMTLossCompute(model.generator, vocab.size, padding_idx,
            label_smoothing=opt.label_smoothing)
    if use_gpu(opt):
        loss.cuda()
    return loss
