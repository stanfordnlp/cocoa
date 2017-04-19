'''
Preprocess examples in a dataset and generate data for models.
'''

import src.config as config
if config.task == config.MutualFriends:
    from mutualfriends.preprocess import *
elif config.task == config.Negotiation:
    from negotiation.preprocess import *
else:
    raise ValueError('Unknown task: %s.' % config.task)



