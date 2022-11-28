import torch
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler


class ReplayBuffer:
    def __init__(self,
                 num_steps,
                 num_processes,
                 obs_size,
                 act_size,
                 norm_rewards=True,
                 ):
        self.obs = torch.zeros(num_steps + 1, num_processes, obs_size)
        self.actions = torch.zeros(num_steps, num_processes, act_size)
        self.rewards = torch.zeros(num_steps, num_processes, 1)
        self.masks = torch.ones(num_steps + 1, num_processes, 1)
        self.returns = torch.zeros(num_steps + 1, num_processes, 1)

        self.value_preds = torch.zeros(num_steps + 1, num_processes, 1)
        self.action_log_probs = torch.zeros(num_steps, num_processes, 1)

        if norm_rewards:
            self.__eps = torch.tensor(1e-6, dtype=torch.float32)
        else:
            self.__eps = None

        self.num_steps = num_steps
        self.step = 0

    def to(self, device):
        self.obs = self.obs.to(device)
        self.actions = self.actions.to(device)
        self.rewards = self.rewards.to(device)
        self.masks = self.masks.to(device)
        self.returns = self.returns.to(device)

        self.value_preds = self.value_preds.to(device)
        self.action_log_probs = self.action_log_probs.to(device)

        if self.__eps:
            self.__eps = self.__eps.to(device)

    def insert(self, obs, actions, rewards, masks):
        self.obs[self.step + 1].copy_(obs)
        self.actions[self.step].copy_(actions)
        self.rewards[self.step].copy_(rewards)
        self.masks[self.step + 1].copy_(masks)

        self.step = (self.step + 1) % self.num_steps

    def update_value_log_prob(self, value, action_log_prob):
        self.value_preds.copy_(value)
        self.action_log_probs.copy_(action_log_prob)

    def after_update(self):
        self.obs[0].copy_(self.obs[-1])
        self.masks[0].copy_(self.masks[-1])

    def compute_returns(self, gamma=0.99, gae_lambda=0.95):
        # Normalize rewards
        if self.__eps:
            mean, std = self.rewards.mean(), self.rewards.std()
            if std > self.__eps:
                self.rewards = (self.rewards - mean) / std

        if gae_lambda:
            gae = 0
            for step in reversed(range(self.rewards.size(0))):
                delta = self.rewards[step] + \
                        gamma * self.value_preds[step + 1] * self.masks[step + 1] - \
                        self.value_preds[step]
                gae = delta + gamma * gae_lambda * self.masks[step + 1] * gae
                self.returns[step] = gae + self.value_preds[step]
        else:
            self.returns[-1] = self.value_preds[-1]
            for step in reversed(range(self.rewards.size(0))):
                self.returns[step] = self.returns[step + 1] * \
                                     gamma * self.masks[step + 1] + self.rewards[step]

    def batch_data_generator(self,
                             advantages,
                             num_mini_batch=None,
                             mini_batch_size=None):
        num_steps, num_processes = self.rewards.size()[0:2]
        batch_size = num_processes * num_steps

        if mini_batch_size is None:
            assert batch_size >= num_mini_batch, (
                "PPO requires the number of processes ({}) "
                "* number of steps ({}) = {} "
                "to be greater than or equal to the number of PPO mini batches ({})."
                "".format(num_processes, num_steps, num_processes * num_steps,
                          num_mini_batch))
            mini_batch_size = batch_size // num_mini_batch
        sampler = BatchSampler(
            SubsetRandomSampler(range(batch_size)),
            mini_batch_size,
            drop_last=True)
        for indices in sampler:
            obs_batch = self.obs[:-1].view(-1, *self.obs.size()[2:])[indices]
            actions_batch = self.actions.view(-1,
                                              self.actions.size(-1))[indices]
            value_preds_batch = self.value_preds[:-1].view(-1, 1)[indices]
            return_batch = self.returns[:-1].view(-1, 1)[indices]
            masks_batch = self.masks[:-1].view(-1, 1)[indices]
            old_action_log_probs_batch = self.action_log_probs.view(-1,
                                                                    1)[indices]
            if advantages is None:
                adv_targ = None
            else:
                adv_targ = advantages.view(-1, 1)[indices]

            yield obs_batch, actions_batch, \
                  value_preds_batch, return_batch, masks_batch, old_action_log_probs_batch, adv_targ