import argparse
import copy
import shutil
from datetime import datetime
from os import path

import numpy as np
import numpy.linalg.linalg
import torch
import yaml
from matplotlib import pyplot as plt
from prompt_toolkit import prompt
from torch import nn
from tqdm import tqdm

# noinspection PyUnresolvedReferences
from dataset.pipa import Annotations  # legacy to correctly load dataset.
from helper import Helper
from continuation_helper import ContinuationHelper
from models.simple import SimpleNet
from models.simpler import SimplerNet
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
        loss = hlpr.attack.compute_blind_loss(model, criterion, batch, attack=hlpr.params.backdoor)
        
        # TODO: ggf. backward pass reduzieren
        loss.backward()
        optimizer.step()

        hlpr.report_training_losses_scales(i, epoch)
        if i == hlpr.params.max_batch_id:
            break
    return


def test(hlpr: Helper, epoch, backdoor=False, return_loss=True, validation=False):
    model = hlpr.task.model
    model.eval()
    hlpr.task.reset_metrics()
    if validation:
        loader = hlpr.task.val_loader
    else:
        loader = hlpr.task.test_loader
    with torch.no_grad():
        for i, data in tqdm(enumerate(loader)):
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
        acc = test(hlpr, epoch, backdoor=False, return_loss=False, validation=True)
        test(hlpr, epoch, backdoor=True, validation=True)
        hlpr.save_model(hlpr.task.model, epoch, acc)
        if hlpr.task.scheduler is not None:
            hlpr.task.scheduler.step(epoch)
    print("#####################")
    print(f"Final test accuracy:")
    acc = test(hlpr, epoch, backdoor=False, return_loss=False, validation=False)
    test(hlpr, epoch, backdoor=True, validation=False)

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
    model_path ="saved_models/good_starting_model/linear_model_cuda/model_last.pt.tar_epoch_149.best"
    model = SimplerNet(100).to(device="cuda")
    checkpoint = torch.load(model_path)    
    model.load_state_dict(checkpoint['state_dict'])
    hlpr.task.model = model

    # 2. Start Continuation procedure
    maintask_losses = []
    backdoor_losses = []
    maintask_acc = []
    backdoor_acc = []

    l1_norms = []

    predictor_lr = 0.0005
    corrector_lr = 0.0005

    predictor_optimizer = torch.optim.SGD(hlpr.task.model.parameters(), lr=predictor_lr, momentum=0.9)
    corrector_optimizer = torch.optim.SGD(hlpr.task.model.parameters(), lr=corrector_lr, momentum=0.9)

    predictor_mt_loss = []
    predictor_bd_loss = []
    predictor_mt_acc = []
    predictor_bd_acc = []

    distances_to_initial_solution = []
    weight_distances = []
    initial_model_weights = []

    # 1.5. Calibrate Continuation
    print(f"Calibrating continuation...")
    for calibration_step in range(20):
        train_step(hlpr, calibration_step, hlpr.task.model, corrector_optimizer,
                   hlpr.task.train_loader, attack=True, predictor_step=False)

    _l, _acc = test(hlpr, 0, backdoor=False, return_loss=True)
    _l_bd, _acc_bd = test(hlpr, 0, backdoor=True, return_loss=True)

    maintask_losses.append(_l)
    maintask_acc.append(_acc)
    backdoor_losses.append(_l_bd)
    backdoor_acc.append(_acc_bd)

    for param in model.parameters():
        initial_model_weights.append(param.clone())

    regularization_val = l1_norm(initial_model_weights)
    l1_norms.append(regularization_val)

    print("Entering continuation loop!!!")
    for continuation_iteration in range(hlpr.params.max_continuation_iterations):

        previous_model_weights = []
        for param in model.parameters():
            previous_model_weights.append(param.clone())
        print(f"Entering continuation iteration {continuation_iteration}")
        # Predictor loop
        for predictor_step in range(hlpr.params.predictor_steps):
            if helper.params.other_direction:
                train_step(hlpr, predictor_step, hlpr.task.model, predictor_optimizer,
                    hlpr.task.train_loader, attack=False, predictor_step=True)
            else:
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

        next_model_weights = []
        for param in model.parameters():
            next_model_weights.append(param.clone())

        distance_to_initial = euclidean_distance(initial_model_weights, next_model_weights)
        weight_distance = euclidean_distance(previous_model_weights, next_model_weights)

        _l, _acc = test(hlpr, 0, backdoor=False, return_loss=True)
        _l_bd, _acc_bd = test(hlpr, 0, backdoor=True, return_loss=True)
        regularization_val = l1_norm(next_model_weights)

        maintask_losses.append(_l)
        maintask_acc.append(_acc)
        backdoor_losses.append(_l_bd)
        backdoor_acc.append(_acc_bd)
        distances_to_initial_solution.append(distance_to_initial)
        weight_distances.append(weight_distance)
        l1_norms.append(regularization_val)

        print(f"Pairwise distance after correction step: {weight_distance}")
        print(f"Weight distance to initial solution: {distance_to_initial}")
        print(f"Regularization: {regularization_val}")

        # 6. Intermediary plot
        if continuation_iteration % hlpr.params.plot_continuation_on_iteration == 0 or continuation_iteration==hlpr.params.max_continuation_iterations-1:
            print(f"Plotting {continuation_iteration} under {model_path}")
            plot_results(hlpr, predictor_lr, corrector_lr,
                         maintask_losses, backdoor_losses,
                         predictor_mt_loss, predictor_bd_loss,
                         maintask_acc, backdoor_acc,
                         weight_distances,
                         distances_to_initial_solution,
                         l1_norms)

        if continuation_iteration % hlpr.params.continuation_checkpoints_at == 0 or continuation_iteration==hlpr.params.max_continuation_iterations-1:
            print(f"Checkpointin model {continuation_iteration} under {model_path}")
            hlpr.save_model(model, continuation_iteration, _acc)

    # 7. Write down all results
    save_continuation_results(helper,
                              maintask_losses,
                              maintask_acc,
                              backdoor_losses,
                              backdoor_acc,
                              predictor_mt_loss,
                              predictor_mt_acc,
                              predictor_bd_loss,
                              predictor_bd_acc,
                              distances_to_initial_solution,
                              weight_distances,
                              l1_norms)


def run_continuation_both_directions(hlpr: Helper):
    # 1. Run pretrained model
    print(f"Loading model...")

    # TODO: find better model
    model_path = "saved_models/good_starting_model/linear_model_cuda/model_last.pt.tar_epoch_149.best"
    model = SimplerNet(100).to(device="cuda")
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['state_dict'])
    hlpr.task.model = model

    from collections import deque
    maintask_losses = deque('')
    backdoor_losses = deque('')
    maintask_acc = deque('')
    backdoor_acc = deque('')

    predictor_mt_loss = deque('')
    predictor_bd_loss = deque('')
    predictor_mt_acc = deque('')
    predictor_bd_acc = deque('')

    distances_to_initial_solution = deque('')
    weight_distances = deque('')
    initial_model_weights = deque('')
    l1_norms = deque('')

    predictor_lr = 0.0001
    corrector_lr = 0.0001

    corrector_optimizer = torch.optim.SGD(hlpr.task.model.parameters(), lr=corrector_lr, momentum=0.9)
    predictor_optimizer = torch.optim.SGD(hlpr.task.model.parameters(), lr=predictor_lr, momentum=0.9)

    # 1.5. Calibrate Continuation
    print(f"Calibrating continuation...")
    for calibration_step in range(30):
        train_step(hlpr, calibration_step, hlpr.task.model, corrector_optimizer,
                   hlpr.task.train_loader, attack=True, predictor_step=False)

    initial_model = copy.deepcopy(hlpr.task.model)
    initial_model_weights = [params.clone() for params in model.parameters()]

    collect_loss_and_accuracy(hlpr, maintask_losses, maintask_acc, backdoor_losses, backdoor_acc)
    collect_regularization(initial_model_weights, l1_norms)

    # 2. Start Continuation
    print("Entering continuation loop!!!")
    backwards = hlpr.params.other_direction
    for continuation_iteration in range(hlpr.params.max_continuation_iterations):
        if continuation_iteration > 1 and hlpr.params.max_continuation_iterations / continuation_iteration == 2 \
                and hlpr.params.both_directions:
            print(f"Going into other direction at iteration {continuation_iteration}")
            model = initial_model
            hlpr.task.model = initial_model
            backwards = not backwards
            predictor_lr = 0.0001
            corrector_lr = 0.0001
            corrector_optimizer = torch.optim.SGD(hlpr.task.model.parameters(), lr=corrector_lr, momentum=0.9)
            predictor_optimizer = torch.optim.SGD(hlpr.task.model.parameters(), lr=predictor_lr, momentum=0.9)

        previous_model_weights = [p.clone() for p in model.parameters()]

        print(f"Entering continuation iteration {continuation_iteration}")
        # Predictor loop
        for predictor_step in range(hlpr.params.predictor_steps):
            if backwards:
                train_step(hlpr, predictor_step, hlpr.task.model, predictor_optimizer,
                           hlpr.task.train_loader, attack=False, predictor_step=True)
            else:
                train_step(hlpr, predictor_step, hlpr.task.model, predictor_optimizer,
                           hlpr.task.train_loader, attack=True, predictor_step=True)

        collect_loss_and_accuracy(hlpr, predictor_mt_loss, predictor_mt_acc, predictor_bd_loss, predictor_bd_acc, backwards)

        # Corrector loop
        for corrector_step in range(hlpr.params.corrector_steps):
            train_step(hlpr, corrector_step, hlpr.task.model, corrector_optimizer,
                       hlpr.task.train_loader, attack=True, predictor_step=False)

        next_model_weights = [params.clone() for params in model.parameters()]

        collect_loss_and_accuracy(hlpr, maintask_losses, maintask_acc, backdoor_losses, backdoor_acc, backwards)
        collect_regularization(next_model_weights, l1_norms)
        collect_distance(initial_model_weights, next_model_weights, distances_to_initial_solution, backwards)
        collect_distance(previous_model_weights, next_model_weights, weight_distances, backwards)

        # 6. Intermediary plot
        if continuation_iteration % hlpr.params.plot_continuation_on_iteration == 0 \
                or continuation_iteration == hlpr.params.max_continuation_iterations - 1:
            print(f"Plotting {continuation_iteration} under {model_path}")
            plot_results(hlpr, predictor_lr, corrector_lr,
                         maintask_losses, backdoor_losses,
                         predictor_mt_loss, predictor_bd_loss,
                         maintask_acc, backdoor_acc,
                         weight_distances,
                         distances_to_initial_solution,
                         l1_norms)

        if continuation_iteration % hlpr.params.continuation_checkpoints_at == 0 \
                or continuation_iteration == hlpr.params.max_continuation_iterations - 1:
            print(f"Checkpointin model {continuation_iteration} under {model_path}")
            hlpr.save_model(model, continuation_iteration)

    # 7. Write down all results
    save_continuation_results(helper,
                              maintask_losses,
                              maintask_acc,
                              backdoor_losses,
                              backdoor_acc,
                              predictor_mt_loss,
                              predictor_mt_acc,
                              predictor_bd_loss,
                              predictor_bd_acc,
                              distances_to_initial_solution,
                              weight_distances,
                              l1_norms)

def collect_loss_and_accuracy(hlpr,
                              maintask_losses, maintask_acc,
                              backdoor_losses, backdoor_acc,
                              backwards: bool = False):
    _l, _acc = test(hlpr, 0, backdoor=False, return_loss=True)
    _l_bd, _acc_bd = test(hlpr, 0, backdoor=True, return_loss=True)

    if not backwards:
        maintask_losses.append(_l)
        maintask_acc.append(_acc)
        backdoor_losses.append(_l_bd)
        backdoor_acc.append(_acc_bd)
    else:
        maintask_losses.appendleft(_l)
        maintask_acc.appendleft(_acc)
        backdoor_losses.appendleft(_l_bd)
        backdoor_acc.appendleft(_acc_bd)

def collect_regularization(weights, l1_norms, backwards: bool = False):
    regularization_val = l1_norm(weights)
    if not backwards:
        l1_norms.append(regularization_val)
    else:
        l1_norms.appendleft(regularization_val)

def collect_distance(weights_a, weights_b, weight_distances, backwards: bool = False):
    weight_distance = euclidean_distance(weights_a, weights_b)
    if not backwards:
        weight_distances.append(weight_distance)
    else:
        weight_distances.appendleft(weight_distance)

def euclidean_distance(params_a, params_b):
    d = 0
    with torch.no_grad():
        for layer_a, layer_b in zip(params_a, params_b):
            d += torch.linalg.norm(layer_a - layer_b)**2
        return float(torch.sqrt(d))

def l1_norm(params):
    return float(sum((abs(p).sum() for p in params)))

def save_continuation_results(helper,
                              mt_loss, mt_acc,
                              backdoor_loss, backdoor_acc,
                              predictor_mt_loss, predictor_mt_acc,
                              predictor_bd_loss, predictor_bd_acc,
                              distance_initial_sol, pairwise_distances,
                              l1_norms):
    import csv
    # dd/mm/YY H:M:S
    root = path.abspath('..')
    dt_string = datetime.now().strftime("%d-%m-%Y--%H-%M-%S")
    output_file_path = f"{helper.params.folder_path}/losses.csv"

    with open(output_file_path, mode='w', newline='') as file:
        # Create a CSV writer object
        writer = csv.writer(file)
        writer.writerow(['Main-Task Loss', 'Main Task Accuracy',
                         'Backdoor Loss', 'Backdoor Accuracy',
                         'Predictor Main Task Loss', 'Predictor Main Task Accuracy',
                         'Predictor Backdoor Loss', 'Predictor Backdoor Accuracy',
                         'Distance Initial Solution', 'Distance Pairwise Solution',
                         'l1 Norms'])

        writer.writerows(zip(mt_loss, mt_acc,
                              backdoor_loss, backdoor_acc,
                              predictor_mt_loss, predictor_mt_acc,
                              predictor_bd_loss, predictor_bd_acc,
                              distance_initial_sol, pairwise_distances,
                              l1_norms))

    logger.info(f"Saved csv results to {output_file_path}")

def  plot_results(hlpr,
                 predictor_lr, corrector_lr,
                 maintask_losses, backdoor_losses,
                 predictor_mt_loss, predictor_bd_loss,
                 maintask_acc, backdoor_acc,
                 weight_distances,
                 distances_to_initial_solution,
                 l1_norms):

    f1 = plt.figure(1)
    plt.scatter(maintask_losses, backdoor_losses, )
    plt.scatter(predictor_mt_loss, predictor_bd_loss, )
    plt.plot(maintask_losses, backdoor_losses)
    #plt.scatter(maintask_losses[starting_point], backdoor_losses[starting_point])
    plt.ylabel("backdoor loss")
    plt.xlabel("main task loss")
    plt.title(f"{hlpr.params.max_continuation_iterations} iterations,\
              {hlpr.params.predictor_steps} predictor steps and lr = {predictor_lr}, \
              {hlpr.params.corrector_steps} corrector steps and lr = {corrector_lr}")
    # plt.plot(maintask_losses, backdoor_losses)
    # plt.plot(corr_mt_loss, corr_bd_loss)
    plt.grid()
    plt.savefig(f'{hlpr.params.folder_path}/plot_loss')

    f2 = plt.figure(2)
    plt.plot(maintask_acc)
    plt.plot(backdoor_acc)
    # plt.scatter(predictor_mt_acc, predictor_bd_acc)
    plt.title(f"{hlpr.params.max_continuation_iterations} iterations,\
              {hlpr.params.predictor_steps} predictor steps and lr = {predictor_lr}, \
              {hlpr.params.corrector_steps} corrector steps and lr = {corrector_lr}")
    plt.grid()
    plt.savefig(f'{hlpr.params.folder_path}/plot_accuracy')

    f3 = plt.figure(3)
    plt.plot(weight_distances)
    plt.ylabel("pairwise euclidean weight distances")
    plt.xlabel("continuation steps")
    plt.grid()
    plt.savefig(f'{hlpr.params.folder_path}/plot_weights')

    f4 = plt.figure(4)
    plt.plot(distances_to_initial_solution)
    plt.ylabel("distance to initial model")
    plt.xlabel("continuation steps")
    plt.grid()
    plt.savefig(f'{hlpr.params.folder_path}/plot_total_weight_distance')
    plt.show()

    f5 = plt.figure(5)
    plt.plot(l1_norms)
    plt.ylabel("l1 norm")
    plt.xlabel("continuation steps")
    plt.grid()
    plt.savefig(f'{hlpr.params.folder_path}/plot_l1_norm')
    plt.show()

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

def swipe_parameters(lrs):
    failed_executions = []
    for lr in lrs:
        with open(args.params) as f:
            params = yaml.load(f, Loader=yaml.FullLoader)

        params['current_time'] = datetime.now().strftime('%b.%d_%H.%M.%S')
        params['commit'] = args.commit
        params['name'] = args.name
        params['lr'] = lr

        helper = Helper(params)
        logger.warning(create_table(params))
        try:
            if helper.params.fl:
                fl_run(helper)
            else:
                if helper.params.continuation:
                    run_continuation(helper)
                else:
                    run(helper)
        except Exception as e:
            print(f"Encountered Exception: {e}")
            failed_executions.append((lr, e))
    print(f"Ran {len(lrs)-len(failed_executions)} out of {len(lrs)} executions.")
    for lr, e in failed_executions:
        print(f"Failed execution {lr} because of: {e}")


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

    SWIPE_THROUGH_PARAMETERS = False
    if SWIPE_THROUGH_PARAMETERS:
        swipe_parameters(lrs=[0.5, 0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001])
        exit()

    helper = Helper(params)
    continuation_helper = ContinuationHelper(params)
    logger.warning(create_table(params))

    try:
        if helper.params.fl:
            fl_run(helper)
        else:
            if helper.params.continuation:
                run_continuation_both_directions(helper)
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
