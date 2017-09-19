import argparse
import numpy as np
from scipy.stats import sem
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from cocoa.core.util import read_pickle
from sessions.rulebased_session import Config

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--trial-db', help='.pkl file of config evaluation results')
    args = parser.parse_args()

    trials = read_pickle(args.trial_db)
    x = []
    margins = []
    success = []
    humanlike = []
    for config, results in trials.iteritems():
        config = Config(*config)
        x.append(config.overshoot)
        num_success = 0
        config_margins = []
        config_humanlike = []
        print config
        for chat_id, obj_values in results.iteritems():
            #if obj_values['scenario_id'] == 'S_gZjYtxgaHsAbvELF':
            #    continue
            if obj_values['margin'] is not None and not np.isnan(obj_values['margin']):
                num_success += 1
                config_margins.append(obj_values['margin'])
            config_humanlike.append(float(obj_values['humanlike']))
        print 'margins:', config_margins
        print 'humanlike:', config_humanlike
        print 'success:', num_success, len(results)
        margins.append((np.mean(config_margins), sem(config_margins)))
        humanlike.append((np.mean(config_humanlike), sem(config_humanlike)))
        success.append(float(num_success) / len(results))
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))
    axes[0].errorbar(x, [m[0] for m in margins], yerr=[m[1] for m in margins], label='margins', fmt='o')
    axes[0].set_ylabel('margins')
    axes[1].errorbar(x, [h[0] for h in humanlike], yerr=[h[1] for h in humanlike], label='humanlike', fmt='o')
    axes[1].set_ylabel('humanlike')
    axes[2].scatter(x, success, label='success_rate', color='r')
    axes[2].set_ylabel('success_rate')
    plt.savefig('trials.png')


