# TODO: refactor mutualfriends/evaluate

import src.config as config
import importlib
task_module = importlib.import_module('.'.join(('src.model', config.task, 'evaluate')))
Evaluator = task_module.Evaluator
pred_to_token = task_module.pred_to_token
