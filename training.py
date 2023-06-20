import argparse
import shutil
from datetime import datetime
from os import path

import numpy as np
import yaml
from matplotlib import pyplot as plt
from prompt_toolkit import prompt
from tqdm import tqdm

# noinspection PyUnresolvedReferences
from dataset.pipa import Annotations  # legacy to correctly load dataset.
from helper import Helper
from continuation_helper import ContinuationHelper
from models.simple import SimpleNet
from tasks.batch import Batch
from utils.utils import *

logger = logging.getLogger('logger')


def train_step(hlpr: Helper, epoch, model, optimizer, train_loader, attack=None, predictor_step=False):
    criterion = hlpr.task.criterion
    model.train()

    for i, data in tqdm(enumerate(train_loader)):
        batch = hlpr.task.get_batch(i, data)

        model.zero_grad()
        loss = hlpr.attack.compute_blind_loss(model, criterion, batch, attack, predictor_step=predictor_step)

        # TODO: ggf. backward pass reduzieren
        loss.backward()
        optimizer.step()

        # TODO: change
        # hlpr.report_training_losses_scales(i, epoch)
        return loss


def train(hlpr: Helper, epoch, model, optimizer, train_loader, attack=None):
    criterion = hlpr.task.criterion
    model.train()
    for i, data in tqdm(enumerate(train_loader)):
        batch = hlpr.task.get_batch(i, data)
        model.zero_grad()
        loss = hlpr.attack.compute_blind_loss(model, criterion, batch, attack)
        
        # TODO: ggf. backward pass reduzieren
        loss.backward()
        optimizer.step()

        hlpr.report_training_losses_scales(i, epoch)
        if i == hlpr.params.max_batch_id:
            break
    return


def test(hlpr: Helper, epoch, backdoor=False, return_loss=True):
    model = hlpr.task.model
    model.eval()
    hlpr.task.reset_metrics()
    print(model.parameters())
    with torch.no_grad():
        for i, data in tqdm(enumerate(hlpr.task.test_loader)):
            batch = hlpr.task.get_batch(i, data)
            if backdoor:
                batch = hlpr.attack.synthesizer.make_backdoor_batch(batch,
                                                                    test=True,
                                                                    attack=True)

            outputs = model(batch.inputs)
            hlpr.task.accumulate_metrics(outputs=outputs, labels=batch.labels)
    
    if return_loss:
        for metric in hlpr.task.metrics:
            if metric.name == "Loss":
                loss = metric.get_main_metric_value()
            if metric.name == "Accuracy":
                acc = metric.get_main_metric_value()
        return loss, acc
    metric = hlpr.task.report_metrics(epoch,
                             prefix=f'Backdoor {str(backdoor):5s}. Epoch: ',
                             tb_writer=hlpr.tb_writer,
                             tb_prefix=f'Test_backdoor_{str(backdoor):5s}')
    return metric


def run(hlpr, attack=None):
    acc = test(hlpr, 0, backdoor=False)
    for epoch in range(hlpr.params.start_epoch,
                       hlpr.params.epochs + 1):
        train(hlpr, epoch, hlpr.task.model, hlpr.task.optimizer,
              hlpr.task.train_loader, attack)
        acc = test(hlpr, epoch, backdoor=False)
        test(hlpr, epoch, backdoor=True)
        hlpr.save_model(hlpr.task.model, epoch, acc)
        if hlpr.task.scheduler is not None:
            hlpr.task.scheduler.step(epoch)

def fl_run(hlpr: Helper):
    for epoch in range(hlpr.params.start_epoch,
                       hlpr.params.epochs + 1):
        run_fl_round(hlpr, epoch)
        metric = test(hlpr, epoch, backdoor=False)
        test(hlpr, epoch, backdoor=True)

        hlpr.save_model(hlpr.task.model, epoch, metric)

def run_continuation(hlpr: Helper):
    # 1. Run pretrained model
    print(f"Loading model...")
    # TODO: find better model
    model_path = "saved_models/good_starting_model/model_last_2.pt.tar"
    model = SimpleNet(100).to(device="mps")
    checkpoint = torch.load(model_path)    
    model.load_state_dict(checkpoint['state_dict'])
    hlpr.task.model = model

    # 2. Start Continuation procedure
    maintask_losses = []
    backdoor_losses = []
    maintask_acc = []
    backdoor_acc = []

    _l, _acc = test(hlpr, 0, backdoor=False, return_loss=True)
    _l_bd, _acc_bd = test(hlpr, 0, backdoor=True, return_loss=True)

    maintask_losses.append(_l)
    maintask_acc.append(_acc)
    backdoor_losses.append(_l_bd)
    backdoor_acc.append(_acc_bd)

    predictor_lr = 0.0001
    corrector_lr = 0.0001

    predictor_optimizer = torch.optim.SGD(hlpr.task.model.parameters(), lr=predictor_lr, momentum=0.9)
    corrector_optimizer = torch.optim.SGD(hlpr.task.model.parameters(), lr=corrector_lr, momentum=0.9)

    predictor_mt_loss = []
    predictor_bd_loss = []
    predictor_mt_acc = []
    predictor_bd_acc = []

    print("Entering continuation loop!!!")
    for continuation_iteration in range(hlpr.params.max_continuation_iterations):
        print(f"Entering continuation iteration {continuation_iteration}")
        # Predictor loop
        for predictor_step in range(hlpr.params.predictor_steps):
            train_step(hlpr, predictor_step, hlpr.task.model, predictor_optimizer,
                    hlpr.task.train_loader, attack=True, predictor_step=True)

        _l, _acc = test(hlpr, 0, backdoor=False, return_loss=True)
        _l_bd, _acc_bd = test(hlpr, 0, backdoor=True, return_loss=True)
        predictor_mt_loss.append(_l)
        predictor_mt_acc.append(_acc)
        predictor_bd_loss.append(_l_bd)
        predictor_bd_acc.append(_acc_bd)


        print(predictor_bd_loss)
        print(predictor_mt_loss)
        print(predictor_bd_acc)
        print(predictor_mt_acc)

        # Corrector loop
        for corrector_step in range(hlpr.params.corrector_steps):
            train_step(hlpr, corrector_step, hlpr.task.model, corrector_optimizer,
                    hlpr.task.train_loader, attack=True, predictor_step=False)

        _l, _acc = test(hlpr, 0, backdoor=False, return_loss=True)
        _l_bd, _acc_bd = test(hlpr, 0, backdoor=True, return_loss=True)

        maintask_losses.append(_l)
        maintask_acc.append(_acc)
        backdoor_losses.append(_l_bd)
        backdoor_acc.append(_acc_bd)
                
        # 6. Save intermediary model
        if continuation_iteration % hlpr.params.save_continuation_on_iteration == 0:
            print(f"theoretically saving {continuation_iteration}")
    
    f1 = plt.figure(1)
    plt.scatter(maintask_losses, backdoor_losses, )
    plt.scatter(predictor_mt_loss, predictor_bd_loss, )
    plt.ylabel("backdoor loss")
    plt.xlabel("main task loss")
    plt.title(f"{hlpr.params.max_continuation_iterations} iterations,\
              {hlpr.params.predictor_steps} predictor steps and lr = {predictor_lr}, \
              {hlpr.params.corrector_steps} corrector steps and lr = {corrector_lr}")
    # plt.plot(maintask_losses, backdoor_losses)
    #plt.plot(corr_mt_loss, corr_bd_loss)

    f2 = plt.figure(2)
    plt.plot(maintask_acc)
    plt.plot(backdoor_acc)
    #plt.scatter(predictor_mt_acc, predictor_bd_acc)
    plt.title(f"{hlpr.params.max_continuation_iterations} iterations,\
              {hlpr.params.predictor_steps} predictor steps and lr = {predictor_lr}, \
              {hlpr.params.corrector_steps} corrector steps and lr = {corrector_lr}")
    
    plt.show()

    # 7. Write down all results
    # TODO: ## Write down all results
    exit()


def apply_correction():
    # TODO
    import random
    rf1 = random.random()
    rf2 = random.random()
    val_acc = random.random()
    return rf1, rf2, val_acc

def save_continuation_results(helper, mt_losses, bd_losses):
    import csv
    # dd/mm/YY H:M:S
    root = path.abspath('..')
    dt_string = datetime.now().strftime("%d-%m-%Y--%H-%M-%S")
    output_file_path = helper.params.folder_path + "losses.csv"

    with open(output_file_path, mode='w', newline='') as file:
        # Create a CSV writer object
        writer = csv.writer(file)
        writer.writerow(['Main-Task Loss', 'Backdoor Loss'])
        writer.writerows(zip(mt_losses, bd_losses))

    # TODO: write helper parameters
    logger.info(f"Saved results to {output_file_path}")


def run_fl_round(hlpr, epoch):
    global_model = hlpr.task.model
    local_model = hlpr.task.local_model

    round_participants = hlpr.task.sample_users_for_round(epoch)
    weight_accumulator = hlpr.task.get_empty_accumulator()

    for user in tqdm(round_participants):
        hlpr.task.copy_params(global_model, local_model)
        optimizer = hlpr.task.make_optimizer(local_model)
        for local_epoch in range(hlpr.params.fl_local_epochs):
            if user.compromised:
                train(hlpr, local_epoch, local_model, optimizer,
                      user.train_loader, attack=True)
            else:
                train(hlpr, local_epoch, local_model, optimizer,
                      user.train_loader, attack=False)
        local_update = hlpr.task.get_fl_update(local_model, global_model)
        if user.compromised:
            hlpr.attack.fl_scale_update(local_update)
        hlpr.task.accumulate_weights(weight_accumulator, local_update)

    hlpr.task.update_global_model(weight_accumulator, global_model)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Backdoors')
    parser.add_argument('--params', dest='params', default='utils/params.yaml')
    parser.add_argument('--name', dest='name', required=True)
    parser.add_argument('--commit', dest='commit',
                        default=get_current_git_hash())

    args = parser.parse_args()

    with open(args.params) as f:
        params = yaml.load(f, Loader=yaml.FullLoader)

    params['current_time'] = datetime.now().strftime('%b.%d_%H.%M.%S')
    params['commit'] = args.commit
    params['name'] = args.name

    helper = Helper(params)
    continuation_helper = ContinuationHelper(params)
    logger.warning(create_table(params))

    try:
        if helper.params.fl:
            fl_run(helper)
        else:
            if helper.params.continuation:
                run_continuation(helper)
            else:
                run(helper)
    except (KeyboardInterrupt):
        if helper.params.log:
            answer = prompt('\nDelete the repo? (y/n): ')
            if answer in ['Y', 'y', 'yes']:
                logger.error(f"Fine. Deleted: {helper.params.folder_path}")
                shutil.rmtree(helper.params.folder_path)
                if helper.params.tb:
                    shutil.rmtree(f'runs/{args.name}')
            else:
                logger.error(f"Aborted training. "
                             f"Results: {helper.params.folder_path}. "
                             f"TB graph: {args.name}")
        else:
            logger.error(f"Aborted training. No output generated.")
