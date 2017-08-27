import src.config as config
if config.task == config.MutualFriends:
    from mutualfriends import *
elif config.task == config.Negotiation:
    from negotiation import *
else:
    raise ValueError('Unknown task: %s.' % config.task)



