from lib.models.policy import Policy


class StruturalPolicy(Policy):
    def __init__(self, cfg_specs, agent):
        super().__init__()
        self.cfg_specs = cfg_specs
        self.type = 'gaussian'
        self.agent = agent

