# ------------------------------------------------------------------------------------------------------------------- #
#   @description: Class AgentStructuralControl
#   @author: by Kangyao Huang
#   @created date: 03.Nov.2022
# ------------------------------------------------------------------------------------------------------------------- #

"""
    AgentStructuralControl is the core for structural control experiments,
    which inherits attributes from AgentPPO. Except for optimization, sampling,
    forward and backward operations, this class also provides some other main
    functions below:
        1. Get training settings by input cfg.
        2. Set environment configs and interactions.
        3. Save checkpoints for training.
"""

