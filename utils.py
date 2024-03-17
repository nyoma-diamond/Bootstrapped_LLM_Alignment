from dataclasses import dataclass

@dataclass
class TrainerArgs:
    epochs: int = 10
    batch_size: int = 32
    mini_batch_size: int = 8
    step_each_objective: bool = False
    out_dir: str = './out'

@dataclass
class QueryArgs:
    target_prefix: str = 'PROMPT: '
    target_postfix: str = '\nRESPONSE: '

    supervisor_prefix: str = 'CONTEXT: A generative AI assistant was asked to respond to the following query: '
    supervisor_infix: str = ' The model provided the following response: '
    supervisor_postfix: str = '\nTASK: Rate the model\'s response above in terms of its {} on a scale from 1 ({}) to 10 ({})' + \
                              '\nRATING: '

TRAINER_DEFAULTS = TrainerArgs()
QUERY_DEFAULTS = QueryArgs()

OBJECTIVES = [
    ('accuracy', 'highly inaccurate', 'highly accurate'),
    ('clarity', 'very unclear', 'very clear'),
    ('conciseness', 'very unconcise', 'very concise'),
    ('safety', 'very harmful', 'very safe'),
    ('relevance', 'completely irrelevant', 'fully relevant')
]
