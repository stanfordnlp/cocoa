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

def report_func(epoch, batch, num_batches, start_time, lr, report_stats):
    """
    This is the user-defined batch-level traing progress
    report function.
    Args:
        epoch(int): current epoch count.
        batch(int): current batch count.
        num_batches(int): total number of batches.
        start_time(float): last report time.
        lr(float): current learning rate.
        report_stats(Statistics): old Statistics instance.
    Returns:
        report_stats(Statistics): updated Statistics instance.
    """
    if batch % opt.report_every == -1 % opt.report_every:
        report_stats.output(epoch, batch + 1, num_batches, start_time)
        if opt.exp_host:
            report_stats.log("progress", experiment, lr)
        if opt.tensorboard:
            report_stats.log_tensorboard("progress", writer, lr, epoch)
        report_stats = onmt.Statistics()

    return report_stats

class Statistics(object):
    """
    Accumulator for loss statistics.
    Currently calculates:

    * accuracy
    * perplexity
    * elapsed time
    """
    def __init__(self, loss=0, n_words=0, n_correct=0):
        self.loss = loss
        self.n_words = n_words
        self.n_correct = n_correct
        self.n_src_words = 0
        self.start_time = time.time()

    def update(self, stat):
        self.loss += stat.loss
        self.n_words += stat.n_words
        self.n_correct += stat.n_correct

    def accuracy(self):
        return 100 * (self.n_correct / self.n_words)

    def ppl(self):
        return math.exp(min(self.loss / self.n_words, 100))

    def elapsed_time(self):
        return time.time() - self.start_time

    def output(self, epoch, batch, n_batches, start):
        """Write out statistics to stdout.

        Args:
           epoch (int): current epoch
           batch (int): current batch
           n_batch (int): total batches
           start (int): start time of epoch.
        """
        t = self.elapsed_time()
        print(("Epoch %2d, %5d/%5d; acc: %6.2f; ppl: %6.2f; " +
               "%3.0f src tok/s; %3.0f tgt tok/s; %6.0f s elapsed") %
              (epoch, batch,  n_batches,
               self.accuracy(),
               self.ppl(),
               self.n_src_words / (t + 1e-5),
               self.n_words / (t + 1e-5),
               time.time() - start))
        sys.stdout.flush()

    def log(self, prefix, experiment, lr):
        t = self.elapsed_time()
        experiment.add_scalar_value(prefix + "_ppl", self.ppl())
        experiment.add_scalar_value(prefix + "_accuracy", self.accuracy())
        experiment.add_scalar_value(prefix + "_tgtper",  self.n_words / t)
        experiment.add_scalar_value(prefix + "_lr", lr)

    def log_tensorboard(self, prefix, writer, lr, epoch):
        t = self.elapsed_time()
        values = {
            "ppl": self.ppl(),
            "accuracy": self.accuracy(),
            "tgtper": self.n_words / t,
            "lr": lr,
        }
        writer.add_scalars(prefix, values, epoch)

