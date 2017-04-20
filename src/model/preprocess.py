'''
Preprocess examples in a dataset and generate data for models.
'''

import src.config as config
import importlib
task_module = importlib.import_module('.'.join(('src.model', config.task)))
add_data_generator_arguments = task_module.add_data_generator_arguments
get_data_generator = task_module.get_data_generator
