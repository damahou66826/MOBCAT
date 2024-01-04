import numpy as np
import torch
import os
from dataset import Dataset, collate_fn
from utils.utils import compute_auc, compute_accuracy, data_split, batch_accuracy
from model import MAMLModel
from policy import PPO, Memory, StraightThrough
from copy import deepcopy
from utils.configuration import create_parser, initialize_seeds
import time
import os
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
DEBUG = False if torch.cuda.is_available() else True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
best_val_score, best_test_score = 0, 0
best_val_auc, best_test_auc = 0, 0
best_epoch = -1

'''
    这个函数的作用是克隆元参数（meta-parameters）并扩展为与批次数据中的标签数量相匹配的形状。
    .clone() 操作用于创建 meta_params[0] 的深拷贝，以确保列表中的每个元素都是独立的张量对象，而不是对原始张量的引用。
    meta_params[0] 他的维度为 [1,1] 或者 [1,question_number] 因此这个操作是将其变为[len(batch['input_labels'], 1] 或者 [len(batch['input_labels'], question_number]  debug已经验证
'''
def clone_meta_params(batch):
    return [meta_params[0].expand(len(batch['input_labels']),  -1).clone(
    )]

'''
    论文中的inner level算法
    这个函数实现了一个内部算法（inner algorithm），用于在元学习（meta-learning）中进行内部循环（inner loop）的参数更新过程。
    将预训练好的模型进行参数更新
'''
def inner_algo(batch, config, new_params, create_graph=False):
    for _ in range(params.inner_loop):
        config['meta_param'] = new_params[0]
        res = model(batch, config)
        loss = res['train_loss']
        grads = torch.autograd.grad(
            loss, new_params, create_graph=create_graph)
        new_params = [(new_params[i] - params.inner_lr*grads[i])
                      for i in range(len(new_params))]
        del grads
    config['meta_param'] = new_params[0]
    return


def get_rl_baseline(batch, config):
    model.pick_sample('random', config)
    new_params = clone_meta_params(batch)
    inner_algo(batch, config, new_params)
    # 不进行梯度更新
    with torch.no_grad():
        output = model(batch, config)['output']
    random_baseline = batch_accuracy(output, batch)
    return random_baseline


def pick_rl_samples(batch, config):
    env_states = model.reset(batch)
    action_mask, train_mask = env_states['action_mask'], env_states['train_mask']
    for _ in range(params.n_query):
        with torch.no_grad():
            state = model.step(env_states)
        if config['mode'] == 'train':
            actions = ppo_policy.policy_old.act(state, memory, action_mask)
        else:
            with torch.no_grad():
                actions = ppo_policy.policy_old.act(state, memory, action_mask)
        action_mask[range(len(action_mask)), actions], train_mask[range(
            len(train_mask)), actions] = 0, 1
        env_states['train_mask'], env_states['action_mask'] = train_mask, action_mask
    # train_mask
    config['train_mask'] = env_states['train_mask']
    return


def run_unbiased(batch, config):
    new_params = clone_meta_params(batch)
    # 这行代码的作用是将 batch['input_mask'] 转移到设备上（例如 GPU），并使用 .clone() 创建了一个张量的深拷贝。深拷贝是为了确保在训练过程中对 config['available_mask'] 的修改不会影响到原始的 batch['input_mask']
    config['available_mask'] = batch['input_mask'].to(device).clone()
    if config['mode'] == 'train':
        random_baseline = get_rl_baseline(batch, config)
    pick_rl_samples(batch, config)
    optimizer.zero_grad()
    meta_params_optimizer.zero_grad()
    inner_algo(batch, config, new_params)
    if config['mode'] == 'train':
        res = model(batch, config)
        loss = res['loss']
        loss.backward()
        optimizer.step()
        meta_params_optimizer.step()
        ####
        final_accuracy = batch_accuracy(res['output'], batch)
        reward = final_accuracy - random_baseline
        memory.rewards.append(reward.to(device))
        ppo_policy.update(memory)
        #
    else:
        with torch.no_grad():
            res = model(batch, config)
    memory.clear_memory()
    return res['output']


def pick_biased_samples(batch, config):
    new_params = clone_meta_params(batch)
    env_states = model.reset(batch)
    action_mask, train_mask = env_states['action_mask'], env_states['train_mask']
    for _ in range(params.n_query):
        with torch.no_grad():
            state = model.step(env_states)
            train_mask = env_states['train_mask']
        if config['mode'] == 'train':
            train_mask_sample, actions = st_policy.policy(state, action_mask)
        else:
            with torch.no_grad():
                train_mask_sample, actions = st_policy.policy(
                    state, action_mask)
        action_mask[range(len(action_mask)), actions] = 0
        # env state train mask should be detached
        env_states['train_mask'], env_states['action_mask'] = train_mask + \
            train_mask_sample.data, action_mask
        if config['mode'] == 'train':
            # loss computation train mask should flow gradient
            config['train_mask'] = train_mask_sample+train_mask
            inner_algo(batch, config, new_params, create_graph=True)
            res = model(batch, config)
            loss = res['loss']
            st_policy.update(loss)
    config['train_mask'] = env_states['train_mask']
    return


def run_biased(batch, config):
    new_params = clone_meta_params(batch)
    if config['mode'] == 'train':
        model.eval()
    pick_biased_samples(batch, config)
    optimizer.zero_grad()
    meta_params_optimizer.zero_grad()
    inner_algo(batch, config, new_params)
    if config['mode'] == 'train':
        model.train()
        optimizer.zero_grad()
        res = model(batch, config)
        loss = res['loss']
        loss.backward()
        optimizer.step()
        meta_params_optimizer.step()
        ####
    else:
        with torch.no_grad():
            res = model(batch, config)
    return res['output']


def run_random(batch, config):
    new_params = clone_meta_params(batch)
    # 将优化器中所有参数的梯度置零的操作。
    meta_params_optimizer.zero_grad()
    if config['mode'] == 'train':
        optimizer.zero_grad()
    ###
    config['available_mask'] = batch['input_mask'].to(device).clone()
    # (batch['input_mask'], seq_len)
    config['train_mask'] = torch.zeros(
        len(batch['input_mask']), params.n_question).long().to(device)

    # Random pick once
    config['meta_param'] = new_params[0]
    if sampling == 'random':
        model.pick_sample('random', config)
        inner_algo(batch, config, new_params)
    if sampling == 'active':
        for _ in range(params.n_query):
            model.pick_sample('active', config)
            inner_algo(batch, config, new_params)

    if config['mode'] == 'train':
        res = model(batch, config)
        loss = res['loss']
        loss.backward()
        optimizer.step()
        meta_params_optimizer.step()
        return
    else:
        with torch.no_grad():
            res = model(batch, config)
        output = res['output']
        return output


def train_model():
    global best_val_auc, best_test_auc, best_val_score, best_test_score, best_epoch
    config['mode'] = 'train'
    # 标记此刻训练为第几个epoch
    config['epoch'] = epoch
    # 将model设置为训练模式
    model.train()
    # repeat 默认情况下为10  [100, 101, 102, 103, 104, 105, 106, 107, 108, 109]
    N = [idx for idx in range(100, 100+params.repeat)]
    for batch in train_loader:
        # Select RL Actions, save in config
        if sampling == 'unbiased':
            run_unbiased(batch, config)
        elif sampling == 'biased':
            run_biased(batch, config)
        else:
            run_random(batch, config)
    # Validation
    val_scores, val_aucs = [], []
    test_scores, test_aucs = [], []
    for idx in N:
        _, auc, acc = pre_test_model(id_=idx, split='val')
        val_scores.append(acc)
        val_aucs.append(auc)
    val_score = sum(val_scores)/(len(N)+1e-20)
    val_auc = sum(val_aucs)/(len(N)+1e-20)

    if best_val_score < val_score:
        best_epoch = epoch
        best_val_score = val_score
        best_val_auc = val_auc
        # Run on test set
        for idx in N:
            _, auc, acc = pre_test_model(id_=idx, split='test')
            test_scores.append(acc)
            test_aucs.append(auc)

        best_test_score = sum(test_scores)/(len(N)+1e-20)
        best_test_auc = sum(test_aucs)/(len(N)+1e-20)
    # 输出不用看
    print('Test_Epoch: {}; val_scores: {}; val_aucs: {}; test_scores: {}; test_aucs: {}'.format(
        epoch, val_scores, val_aucs, test_scores, test_aucs))
    if params.neptune:
        neptune.log_metric('Valid Accuracy', val_score)
        neptune.log_metric('Best Test Accuracy', best_test_score)
        neptune.log_metric('Best Test Auc', best_test_auc)
        neptune.log_metric('Best Valid Accuracy', best_val_score)
        neptune.log_metric('Best Valid Auc', best_val_auc)
        neptune.log_metric('Best Epoch', best_epoch)
        neptune.log_metric('Epoch', epoch)


def pre_test_model(id_, split='val'):
    model.eval()  # 用于将模型切换到评估模式的方法。在深度学习中，模型在训练和评估阶段有不同的行为。在训练阶段，模型通常会执行前向传播、计算损失和反向传播等操作，并根据损失函数进行参数更新。而在评估阶段，模型主要用于推断和预测，不需要进行参数更新。
    config['mode'] = 'test'
    if split == 'val':
        valid_dataset.seed = id_
    elif split == 'test':
        test_dataset.seed = id_
    loader = torch.utils.data.DataLoader(
        valid_dataset if split == 'val' else test_dataset, collate_fn=collate_fn, batch_size=params.test_batch_size, num_workers=num_workers, shuffle=False, drop_last=False)

    total_loss, all_preds, all_targets = 0., [], []
    n_batch = 0
    for batch in loader:
        if sampling == 'unbiased':
            output = run_unbiased(batch, config)
        elif sampling == 'biased':
            output = run_biased(batch, config)
        else:
            output = run_random(batch, config)             # 预测值
        target = batch['output_labels'].float().numpy()    # 真实值
        mask = batch['output_mask'].numpy() == 1
        all_preds.append(output[mask])
        all_targets.append(target[mask])
        n_batch += 1

    all_pred = np.concatenate(all_preds, axis=0)
    all_target = np.concatenate(all_targets, axis=0)
    auc = compute_auc(all_target, all_pred)
    accuracy = compute_accuracy(all_target, all_pred)
    return total_loss/n_batch, auc, accuracy


if __name__ == "__main__":
    params = create_parser()
    print(params)
    if params.use_cuda:
        assert device.type == 'cuda', 'no gpu found!'

    if params.neptune:
        import neptune
        project = "arighosh/bobcat"
        neptune.init(project_qualified_name=project,
                     api_token=os.environ["NEPTUNE_API_TOKEN"])
        neptune_exp = neptune.create_experiment(
            name=params.file_name, send_hardware_metrics=False, params=vars(params))

    config = {}
    initialize_seeds(params.seed)

    #
    base, sampling = params.model.split('-')[0], params.model.split('-')[-1]
    if base == 'biirt':
        model = MAMLModel(sampling=sampling, n_query=params.n_query,
                          n_question=params.n_question, question_dim=1).to(device)
        # 这行代码的目的是创建一个包含一个元素的列表 meta_params，其中这个元素是一个形状为 (1, 1) 的张量，其值是从均值为 -1、标准差为 1 的正态分布中随机采样的。此张量被移动到指定的设备（device）上，并设置为需要梯度计算。
        meta_params = [torch.Tensor(
            1, 1).normal_(-1., 1.).to(device).requires_grad_()]
    if base == 'binn':
        model = MAMLModel(sampling=sampling, n_query=params.n_query,
                          n_question=params.n_question, question_dim=params.question_dim).to(device)
        # 这行代码的目的是创建一个包含一个元素的列表 meta_params，其中这个元素是一个形状为 (1, question_dim) 的张量，其值是从均值为 -1、标准差为 1 的正态分布中随机采样的。此张量被移动到指定的设备（device）上，并设置为需要梯度计算。
        meta_params = [torch.Tensor(
            1, params.question_dim).normal_(-1., 1.).to(device).requires_grad_()]

    optimizer = torch.optim.Adam(
        model.parameters(), lr=params.lr, weight_decay=1e-8)
    meta_params_optimizer = torch.optim.SGD(
        meta_params, lr=params.meta_lr, weight_decay=2e-6, momentum=0.9)
    if params.neptune:
        neptune_exp.log_text(
            'model_summary', repr(model))
    print(model)

    #
    if sampling == 'unbiased':
        betas = (0.9, 0.999)
        K_epochs = 4                # update policy for K epochs
        eps_clip = 0.2              # clip parameter for PPO
        memory = Memory()
        ppo_policy = PPO(params.n_question, params.n_question,
                         params.policy_lr, betas, K_epochs, eps_clip)
        if params.neptune:
            neptune_exp.log_text(
                'ppo_model_summary', repr(ppo_policy.policy))
    if sampling == 'biased':
        betas = (0.9, 0.999)
        st_policy = StraightThrough(params.n_question, params.n_question,
                                    params.policy_lr, betas)
        if params.neptune:
            neptune_exp.log_text(
                'biased_model_summary', repr(st_policy.policy))

    # 数据处理
    data_path = os.path.normpath('data/train_task_'+params.dataset+'.json')
    train_data, valid_data, test_data = data_split(
        data_path, params.fold,  params.seed)
    train_dataset, valid_dataset, test_dataset = Dataset(
        train_data), Dataset(valid_data), Dataset(test_data)
    # 训练模型
    # 用于数据加载的子进程数量
    num_workers = 3
    collate_fn = collate_fn(params.n_question)
    '''
        这段代码使用PyTorch的DataLoader来创建一个用于训练的数据加载器 (train_loader)。让我们逐步解释每个参数的含义：

        train_dataset: 这是一个PyTorch数据集对象，包含用于训练的样本。
        collate_fn: 这是一个用于处理每个batch的函数，可以根据需要自定义。通常，它用于将一个batch的样本整理成一个可以输入到模型的形式。
        batch_size: 这是每个batch中包含的样本数。
        num_workers: 这是用于数据加载的子进程数。它可以加速数据加载，尤其是在数据预处理较为耗时时。
        shuffle=True: 这表示在每个epoch开始时是否对数据进行洗牌，即打乱样本的顺序，以确保模型在每个epoch中看到不同的样本顺序，有助于更好的训练。
        drop_last=True: 如果数据样本总数不能被batch size整除，设置为True时会删除最后一个无法构成完整batch的样本。
    '''
    train_loader = torch.utils.data.DataLoader(
        train_dataset, collate_fn=collate_fn, batch_size=params.train_batch_size, num_workers=num_workers, shuffle=True, drop_last=True)
    start_time = time.time()
    for epoch in range(params.n_epoch):
        train_model()
        if epoch >= (best_epoch+params.wait):
            break
