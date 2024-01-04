import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from copy import deepcopy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.action_masks = []

    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.action_masks[:]


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, n_latent_var=256):
        super(ActorCritic, self).__init__()

        # actor
        self.obs_layer = nn.Linear(state_dim, n_latent_var)
        self.actor_layer = nn.Sequential(
            nn.Linear(n_latent_var, n_latent_var),
            nn.Tanh(),
            nn.Linear(n_latent_var, action_dim)
        )

        #self.action_layer_weight = nn.Parameter(torch.ones(1,state_dim))
        #self.action_layer_bias = nn.Parameter(torch.zeros(1, state_dim))
        # critic
        self.value_layer = nn.Sequential(
            nn.Linear(n_latent_var, n_latent_var),
            nn.Tanh(),
            nn.Linear(n_latent_var, 1)
        )

    def forward(self):
        raise NotImplementedError

    def hidden_state(self, state):
        hidden_state = self.obs_layer(state)
        return hidden_state

    def act(self, state, memory, action_mask):
        # self.action_layer_weight * state + self.action_layer_bias #B, N
        hidden_state = self.hidden_state(state)
        logits = self.actor_layer(hidden_state)
        inf_mask = torch.clamp(torch.log(action_mask.float()),
                               min=torch.finfo(torch.float32).min)
        logits = logits + inf_mask
        action_probs = F.softmax(logits, dim=-1)
        dist = Categorical(action_probs)
        action = dist.sample()

        memory.states.append(state)
        memory.actions.append(action)
        memory.action_masks.append(deepcopy(action_mask))
        memory.logprobs.append(dist.log_prob(action))

        return action.detach()

    def evaluate(self, state, action, action_mask):
        hidden_state = self.hidden_state(state)
        logits = self.actor_layer(hidden_state)
        inf_mask = torch.clamp(torch.log(action_mask.float()),
                               min=torch.finfo(torch.float32).min)
        logits = logits + inf_mask
        action_probs = F.softmax(logits, dim=-1)
        # 你可以使用 dist.sample() 来从概率分布中采样一个动作
        dist = Categorical(action_probs)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()

        state_value = self.value_layer(hidden_state)

        return action_logprobs, torch.squeeze(state_value), dist_entropy

'''
    PPO proximal Policy Optimization 
    unbiased Gradient Estimate
    proximal policy optimization
    利用了actor 网络 与 critic网络 来保证训练的稳定性
'''
class PPO:
    def __init__(self, state_dim, action_dim, lr, betas, K_epochs, eps_clip):
        self.lr = lr
        self.betas = betas
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs

        self.policy = ActorCritic(state_dim, action_dim).to(device)
        self.optimizer = torch.optim.Adam(
            self.policy.parameters(), lr=lr, betas=betas)
        self.policy_old = ActorCritic(state_dim, action_dim).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss = nn.MSELoss()

    def update(self, memory):
        # Monte Carlo estimate of state rewards:
        rewards = []
        # discounted_reward = 0
        # for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
        #     if is_terminal:
        #         discounted_reward = 0
        #     discounted_reward = reward + (self.gamma * discounted_reward)
        #     rewards.insert(0, discounted_reward)

        # Normalizing the rewards:
        #rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        rewards = memory.rewards[0].repeat(len(memory.actions))
        rewards = rewards / (rewards.std() + 1e-5)
        #rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)

        # convert list to tensor
        old_states = torch.cat(memory.states, dim=0).detach()
        old_actions = torch.cat(memory.actions, dim=0).detach()
        old_logprobs = torch.cat(memory.logprobs, dim=0).detach()
        old_actionmask = torch.cat(memory.action_masks, dim=0).detach()

        # Optimize policy for K epochs:
        for _ in range(self.K_epochs):
            # Evaluating old actions and values :
            logprobs, state_values, dist_entropy = self.policy.evaluate(
                old_states, old_actions, old_actionmask)

            # Finding the ratio (pi_theta / pi_theta__old):
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss:
            advantages = rewards - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip,
                                1+self.eps_clip) * advantages
            loss = -torch.min(surr1, surr2) + 0.5 * \
                self.MseLoss(state_values, rewards) - 0.01*dist_entropy

            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        # Copy new weights into old policy:
        self.policy_old.load_state_dict(self.policy.state_dict())

'''
    hard_sample 为 Actor所用
    logit ： action_dim
'''
def hard_sample(logits, dim=-1):
    # dim 参数用于指定 softmax 操作沿着哪个维度进行计算。dim=-1 表示 softmax 操作将在张量的最后一个维度上进行，也就是张量的最内层维度
    # 例如，如果 logits 是一个二维张量，那么 dim=-1 将 softmax 操作应用于每一行，使每一行的元素之和等于1。
    y_soft = F.softmax(logits, dim=-1)
    # 这代码使用了 PyTorch 中的 max 函数，dim 参数用于指定沿哪个维度进行最大值的计算。keepdim=True 表示在结果中保持原始张量的维度。
    #
    # 具体而言，y_soft.max(dim, keepdim=True) 返回一个元组 (values, indices)，其中 values 包含沿指定维度 dim 的最大值，而 indices 包含相应位置的索引。
    #
    # index = y_soft.max(dim, keepdim=True)[1] 就提取了这个元组中的索引部分，即标记了在每个位置上取得最大值的索引。这通常用于获取最大概率对应的类别。
    index = y_soft.max(dim, keepdim=True)[1]
    # y_hard 的初始化是与 y_soft 大小相同的零张量。然后，scatter_ 函数通过 index 参数，将在 dim 维度上对应索引位置的值设置为 1.0，其他位置保持为零。
    #
    # 这样的操作实际上是将概率分布转化为一个独热编码，其中概率最大的位置对应的元素值为 1，其他位置为 0。
    y_hard = torch.zeros_like(y_soft).scatter_(dim, index, 1.0)
    # 具体而言，y_hard - y_soft.detach() 计算了 y_hard 与 y_soft 之间的差异。由于 y_soft 上的 .detach() 操作，这里计算的差异不会影响到梯度反向传播，因为 detach() 将 y_soft 从计算图中分离。
    ret = y_hard - y_soft.detach() + y_soft
    # 如果 index 的形状是 (batch_size, 1)，那么执行 index.squeeze(1) 后，得到的张量将是形状 (batch_size,)，去除了原来的维度为 1 的那一维。这样的操作常常用于去除不必要的维度，以便与其他张量进行操作或计算。
    return ret, index.squeeze(1)

'''
    输入 ： 状态state
    输出 ： 行为action
'''
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, n_latent_var=256):
        super().__init__()
        # actor  输入 state_dim 维度，输出n_latent_var维度
        self.obs_layer = nn.Linear(state_dim, n_latent_var)
        self.actor_layer = nn.Sequential(
            nn.Linear(n_latent_var, n_latent_var),
            nn.Tanh(),
            nn.Linear(n_latent_var, action_dim)
        )

    def forward(self, state, action_mask):
        hidden_state = self.obs_layer(state)
        logits = self.actor_layer(hidden_state)
        # torch.clamp 对向量中的值进行截断操作，这里表示使得里面的值每个都大于 32 位浮点数类型（float32）的最小正子规范化数。这个值表示能够在 float32 类型中表示的最小的正数
        # 32 位浮点数的最小正子规范化数是 1.0 * 2^(-126)，对应于 Python 中的表示方式是 2.3509887e-38
        inf_mask = torch.clamp(torch.log(action_mask.float()),
                               min=torch.finfo(torch.float32).min)
        logits = logits + inf_mask
        train_mask, actions = hard_sample(logits)
        return train_mask, actions

'''
    approximate gradient estimate
'''
class StraightThrough:
    def __init__(self, state_dim, action_dim, lr, betas):
        self.lr = lr
        self.betas = betas
        self.policy = Actor(state_dim, action_dim).to(device)
        self.optimizer = torch.optim.Adam(
            self.policy.parameters(), lr=lr, betas=betas)
    '''
        update 方法中实现了更新策略网络参数的过程。具体步骤如下：

        self.optimizer.zero_grad()：清零梯度，防止梯度累积。
        loss.mean().backward()：计算损失的均值并进行反向传播。这里的损失是一个 PyTorch 张量，通过调用 backward 方法，计算图中的梯度。
        self.optimizer.step()：根据梯度更新策略网络的参数。
    '''
    def update(self, loss):
        self.optimizer.zero_grad()
        loss.mean().backward()
        self.optimizer.step()


def main():
    pass


if __name__ == '__main__':
    main()
