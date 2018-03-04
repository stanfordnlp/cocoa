import onmt
from onmt.Utils import use_gpu

def make_loss(opt, tgt_vocab, model):
    loss = onmt.Loss.NMTLossCompute(
            model.generator, tgt_vocab,
            label_smoothing=opt.label_smoothing)
    if use_gpu(opt):
        loss.cuda()
    return loss
