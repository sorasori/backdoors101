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
from collections import deque

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

        model.backward_passes += 1
        loss.backward()
        optimizer.step()
        return loss

# TODO: implement this
#    t = time.perf_counter()
#    record_time(params, t, 'forward')
#    t = time.perf_counter()
#    record_time(params, t, 'backward')


def train(hlpr: Helper, epoch, model, optimizer, train_loader, attack=None):
    criterion = hlpr.task.criterion
    model.train()
    for i, data in tqdm(enumerate(train_loader)):
        batch = hlpr.task.get_batch(i, data)
        model.zero_grad()
        loss = hlpr.attack.compute_blind_loss(model, criterion, batch, attack=hlpr.params.backdoor)
        
        loss.backward()
        optimizer.step()
        hlpr.report_training_losses_scales(i, epoch)
        if i == hlpr.params.max_batch_id:
            break
    return


def test(hlpr: Helper, epoch, backdoor=False, return_loss=True, validation=False, train=False):
    model = hlpr.task.model
    model.eval()
    hlpr.task.reset_metrics()
    if validation:
        loader = hlpr.task.val_loader
    elif train:
        loader = hlpr.task.train_loader
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

    train_accs_mt = []
    train_losses_mt = []
    val_accs_mt = []
    val_losses_mt = []

    train_accs_bd = []
    train_losses_bd = []
    val_accs_bd = []
    val_losses_bd = []

    val_loss, val_acc = test(hlpr, 0, backdoor=False, return_loss=True, validation=True)
    val_accs_mt.append(val_acc)
    val_losses_mt.append(val_loss)

    train_loss, train_acc = test(hlpr, 0, backdoor=False, return_loss=True, validation=False, train=True)
    train_accs_mt.append(train_acc)
    train_losses_mt.append(train_loss)

    val_loss_bd, val_acc_bd,  = test(hlpr, 0, backdoor=True, return_loss=True, validation=True)
    val_accs_bd.append(val_acc_bd)
    val_losses_bd.append(val_loss_bd)

    train_loss_bd, train_acc_bd = test(hlpr, 0, backdoor=True, return_loss=True, validation=False, train=True)
    train_accs_bd.append(train_acc_bd)
    train_losses_bd.append(train_loss_bd)

    for epoch in range(1,
                       hlpr.params.epochs + 1):
        train(hlpr, epoch, hlpr.task.model, hlpr.task.optimizer,
              hlpr.task.train_loader, attack)

        val_loss, val_acc = test(hlpr, epoch, backdoor=False, return_loss=True, validation=True)
        val_accs_mt.append(val_acc)
        val_losses_mt.append(val_loss)

        train_loss, train_acc = test(hlpr, epoch, backdoor=False, return_loss=True, validation=False, train=True)
        train_accs_mt.append(train_acc)
        train_losses_mt.append(train_loss)

        val_loss_bd, val_acc_bd, = test(hlpr, epoch, backdoor=True, return_loss=True, validation=True)
        val_accs_bd.append(val_acc_bd)
        val_losses_bd.append(val_loss_bd)

        train_loss_bd, train_acc_bd = test(hlpr, epoch, backdoor=True, return_loss=True, validation=False, train=True)
        train_accs_bd.append(train_acc_bd)
        train_losses_bd.append(train_loss_bd)

        # test(hlpr, epoch, backdoor=True, return_loss=False, validation=True)
        hlpr.save_model(hlpr.task.model, epoch, val_acc)
        if hlpr.task.scheduler is not None:
            hlpr.task.scheduler.step(epoch)
    
    print("#####################")
    print(f"Final test accuracy:")
    acc = test(hlpr, epoch, backdoor=False, return_loss=False, validation=False)
    test(hlpr, epoch, backdoor=True, validation=False)

    write_training_data_to_csv(hlpr.params.epochs,
                               train_losses_mt, train_accs_mt,
                               train_losses_bd, train_accs_bd,
                               val_losses_mt, val_accs_mt,
                               val_losses_bd, val_accs_bd
                               )


def write_training_data_to_csv(epochs,
                               train_losses_mt, train_accs_mt,
                               train_losses_bd, train_accs_bd, 
                               val_losses_mt, val_accs_mt,
                               val_losses_bd, val_accs_bd) -> None:
    import csv
    output_file_path = f"{helper.params.folder_path}/training_losses.csv"

    with open(output_file_path, mode='w', newline='') as file:
        # Create a CSV writer object
        writer = csv.writer(file)
        writer.writerow(["epoch",
                         'Main Task Training Loss', 'Main Task Training Accuracy',
                         'Backdoor Training Loss', 'Backdoor Training Accuracy',
                         'Main Task Validation Loss', 'Main Task Validation Accuracy',
                         'Backdoor Validation Loss', 'Backdoor Validation Accuracy'
                         ])

        writer.writerows(zip(range(0, epochs),
                               train_losses_mt, train_accs_mt,
                               train_losses_bd, train_accs_bd, 
                               val_losses_mt, val_accs_mt,
                               val_losses_bd, val_accs_bd))

    logger.info(f"Saved csv results to {output_file_path}")

    

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
    model_path = hlpr.params.continuation_model_path
    model = SimpleNet(100).to(device="cuda")
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['state_dict'])
    hlpr.task.model = model

    maintask_losses = deque('')
    backdoor_losses = deque('')
    maintask_acc = deque('')
    backdoor_acc = deque('')
    predictor_mt_loss = deque('')
    predictor_bd_loss = deque('')
    predictor_mt_acc = deque('')
    predictor_bd_acc = deque('')

    if hlpr.params.track_training:
        tr_maintask_losses = deque('')
        tr_backdoor_losses = deque('')
        tr_maintask_acc = deque('')
        tr_backdoor_acc = deque('')
        tr_predictor_mt_loss = deque('')
        tr_predictor_bd_loss = deque('')
        tr_predictor_mt_acc = deque('')
        tr_predictor_bd_acc = deque('')

    distances_to_initial_solution = deque('')
    weight_distances = deque('')
    initial_model_weights = deque('')
    l1_norms = deque('')

    predictor_lr = hlpr.params.predictor_lr
    corrector_lr = hlpr.params.corrector_lr

    if hlpr.params.optimizer == "SGD":
        corrector_optimizer = torch.optim.SGD(hlpr.task.model.parameters(), lr=corrector_lr, momentum=0.9)
        predictor_optimizer = torch.optim.SGD(hlpr.task.model.parameters(), lr=predictor_lr, momentum=0.9)
    else:
        corrector_optimizer = torch.optim.Adam(hlpr.task.model.parameters(), lr=corrector_lr)
        predictor_optimizer = torch.optim.Adam(hlpr.task.model.parameters(), lr=predictor_lr)

    forward_passes_per_pred = 0
    forward_passes_per_corr = 0
    backward_passes_per_pred = 0
    backward_passes_per_corr = 0
    sum_forward_passes = 0
    sum_backward_passes = 0

    # 1.5. Calibrate Continuation
    print(f"Calibrating continuation...")
    if hlpr.params.calibration:
        for cal_step in range(hlpr.params.calibration_iterations):
            print(f"Calibration iteration: {cal_step}")
            train_step(hlpr, cal_step, hlpr.task.model, corrector_optimizer,
                    hlpr.task.train_loader, attack=True, predictor_step=False)
        print(f"Checkpointing calibrated model under {model_path}")
        hlpr.save_model(model, -1)

    initial_model = copy.deepcopy(hlpr.task.model)
    initial_model_weights = [params.clone() for params in model.parameters()]

    collect_loss_and_accuracy(hlpr, maintask_losses, maintask_acc, backdoor_losses, backdoor_acc)
    if hlpr.params.track_training:
        collect_loss_and_accuracy(hlpr, tr_maintask_losses, tr_maintask_acc, tr_backdoor_losses, tr_backdoor_acc, training=True)
    collect_regularization(initial_model_weights, l1_norms)

    # 2. Start Continuation
    print("Entering continuation loop!!!")
    backwards = hlpr.params.other_direction
    start_time = time.time()
    for continuation_iteration in range(hlpr.params.max_continuation_iterations):
        print(f"Entering continuation iteration {continuation_iteration} at {time.strftime('%H:%M:%S', time.gmtime(time.time() - start_time))}")
        if continuation_iteration > 1 and hlpr.params.max_continuation_iterations / continuation_iteration == 2 \
                and hlpr.params.both_directions:
            print(f"Going into other direction at iteration {continuation_iteration}")
            model = initial_model
            hlpr.task.model = initial_model
            backwards = not backwards
            corrector_optimizer = torch.optim.SGD(hlpr.task.model.parameters(), lr=corrector_lr, momentum=0.9)
            predictor_optimizer = torch.optim.SGD(hlpr.task.model.parameters(), lr=predictor_lr, momentum=0.9)
        previous_model_weights = [p.clone() for p in model.parameters()]

        print(f"Entering continuation iteration {continuation_iteration}")
        # Predictor loop
        model.reset_passes()
        for predictor_step in range(hlpr.params.predictor_steps):
            if backwards:
                train_step(hlpr, predictor_step, hlpr.task.model, predictor_optimizer,
                           hlpr.task.train_loader, attack=False, predictor_step=True)
            else:
                train_step(hlpr, predictor_step, hlpr.task.model, predictor_optimizer,
                           hlpr.task.train_loader, attack=True, predictor_step=True)
        print(f"Finished predictor step at {time.strftime('%H:%M:%S', time.gmtime(time.time() - start_time))}")
        if continuation_iteration <= 1:
            forward_passes_per_pred = model.forward_passes
            backward_passes_per_pred = model.backward_passes
        sum_forward_passes += model.forward_passes
        sum_backward_passes += model.backward_passes

        collect_loss_and_accuracy(hlpr, predictor_mt_loss, predictor_mt_acc, predictor_bd_loss, predictor_bd_acc, backwards)
        if hlpr.params.track_training:
            collect_loss_and_accuracy(hlpr, tr_predictor_mt_loss, tr_predictor_mt_acc, tr_predictor_bd_loss, tr_predictor_bd_acc, backwards, training=True)

        # Corrector loop
        model.reset_passes()
        for corrector_step in range(hlpr.params.corrector_steps):
            train_step(hlpr, corrector_step, hlpr.task.model, corrector_optimizer,
                       hlpr.task.train_loader, attack=True, predictor_step=False)
        print(f"Finished corrector step at {time.strftime('%H:%M:%S', time.gmtime(time.time() - start_time))}")
        if continuation_iteration == 1:
            forward_passes_per_corr = model.forward_passes
            backward_passes_per_corr = model.backward_passes
        sum_forward_passes += model.forward_passes
        sum_backward_passes += model.backward_passes

        next_model_weights = [params.clone() for params in model.parameters()]

        collect_loss_and_accuracy(hlpr, maintask_losses, maintask_acc, backdoor_losses, backdoor_acc, backwards)
        if hlpr.params.track_training:    
            collect_loss_and_accuracy(hlpr, tr_maintask_losses, tr_maintask_acc, tr_backdoor_losses, tr_backdoor_acc,
                                    training=True)
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
    print(f"Total forward passes: {sum_forward_passes}")
    print(f"Total backward passes: {sum_backward_passes}")
    print(f"Forward passes per predictor step: {forward_passes_per_pred}")
    print(f"Forward passes per corrector step: {forward_passes_per_corr}")
    print(f"Backward passes per predictor step: {backward_passes_per_pred}")
    print(f"Backward passes per corrector step: {backward_passes_per_corr}")

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
    if hlpr.params.track_training:
        save_training_continuation_results(helper,
                              tr_maintask_losses,
                              tr_maintask_acc,
                              tr_backdoor_losses,
                              tr_backdoor_acc,
                              tr_predictor_mt_loss,
                              tr_predictor_mt_acc,
                              tr_predictor_bd_loss,
                              tr_predictor_bd_acc,)    


def collect_loss_and_accuracy(hlpr,
                              maintask_losses, maintask_acc,
                              backdoor_losses, backdoor_acc,
                              backwards: bool = False,
                              training: bool = False):
    _l, _acc = test(hlpr, 0, backdoor=False, return_loss=True, validation=False, train=training)
    _l_bd, _acc_bd = test(hlpr, 0, backdoor=True, return_loss=True, validation=False, train=training)

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


def save_training_continuation_results(helper,
                              tr_maintask_losses,
                              tr_maintask_acc,
                              tr_backdoor_losses,
                              tr_backdoor_acc,
                              tr_predictor_mt_loss,
                              tr_predictor_mt_acc,
                              tr_predictor_bd_loss,
                              tr_predictor_bd_acc,):
    import csv
    output_file_path = f"{helper.params.folder_path}/training_losses.csv"
    with open(output_file_path, mode='w', newline='') as file:
        # Create a CSV writer object
        writer = csv.writer(file)
        writer.writerow(['Main-Task Loss', 'Main Task Accuracy',
                         'Backdoor Loss', 'Backdoor Accuracy',
                         'Predictor Main Task Loss', 'Predictor Main Task Accuracy',
                         'Predictor Backdoor Loss', 'Predictor Backdoor Accuracy'])
        writer.writerows(zip(tr_maintask_losses, tr_maintask_acc,
                              tr_backdoor_losses, tr_backdoor_acc,
                              tr_predictor_mt_loss, tr_predictor_mt_acc,
                              tr_predictor_bd_loss, tr_predictor_bd_acc))
    logger.info(f"Saved csv results to {output_file_path}")

def  plot_results(hlpr,
                 predictor_lr, corrector_lr,
                 maintask_losses, backdoor_losses,
                 predictor_mt_loss, predictor_bd_loss,
                 maintask_acc, backdoor_acc,
                 weight_distances,
                 distances_to_initial_solution,
                 l1_norms):
    resolution_value = 2000
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
    plt.grid()
    plt.savefig(f'{hlpr.params.folder_path}/plot_loss', dpi=resolution_value)
    plt.close()

    f2 = plt.figure(2)
    plt.plot(maintask_acc)
    plt.plot(backdoor_acc)
    # plt.scatter(predictor_mt_acc, predictor_bd_acc)
    plt.title(f"{hlpr.params.max_continuation_iterations} iterations,\
              {hlpr.params.predictor_steps} predictor steps and lr = {predictor_lr}, \
              {hlpr.params.corrector_steps} corrector steps and lr = {corrector_lr}")
    plt.grid()
    plt.savefig(f'{hlpr.params.folder_path}/plot_accuracy', dpi=resolution_value)
    plt.close()

    f3 = plt.figure(3)
    plt.plot(weight_distances)
    plt.ylabel("pairwise euclidean weight distances")
    plt.xlabel("continuation steps")
    plt.grid()
    plt.savefig(f'{hlpr.params.folder_path}/plot_weights', dpi=resolution_value)
    plt.close()

    f4 = plt.figure(4)
    plt.plot(distances_to_initial_solution)
    plt.ylabel("distance to initial model")
    plt.xlabel("continuation steps")
    plt.grid()
    plt.savefig(f'{hlpr.params.folder_path}/plot_total_weight_distance', dpi=resolution_value)
    plt.show()
    plt.close()

    f5 = plt.figure(5)
    plt.plot(l1_norms)
    plt.ylabel("l1 norm")
    plt.xlabel("continuation steps")
    plt.grid()
    plt.savefig(f'{hlpr.params.folder_path}/plot_l1_norm', dpi=resolution_value)
    plt.show()
    plt.close()

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


def scalar_loss_fn(weights, loss, out_batch, target_batch, attack_portion):
    res = 0
    res += weights[0] * loss(out_batch[:attack_portion], target_batch.labels[:attack_portion])
    res += weights[1] * loss(out_batch[attack_portion:], target_batch.labels[attack_portion:])
    return res

def train_scalarized(hlpr: Helper, epoch, model, optimizer, train_loader, attack=None, weights=[0.5, 0.5]):
    loss_f = nn.CrossEntropyLoss(reduction='mean')
    model.train()
    for i, data in tqdm(enumerate(train_loader)):
        batch = hlpr.task.get_batch(i, data)
        model.zero_grad()

        # Create poisoned batch
        batch = batch.clip(hlpr.params.clip_batch)
        attack_portion = round(
            batch.batch_size * hlpr.params.poisoning_proportion)
        back_batch = batch.clone()
        hlpr.synthesizer.synthesize_inputs(batch=back_batch, attack_portion=attack_portion)
        hlpr.synthesizer.synthesize_labels(batch=back_batch, attack_portion=attack_portion)

        outputs = model(back_batch.inputs)

        loss = scalar_loss_fn(weights, loss_f, outputs, back_batch, attack_portion)
        loss.backward()
        model.backward_passes += 1
        optimizer.step()
    return


def run_scalarization(hlpr: Helper):
    model_path = hlpr.params.continuation_model_path

    maintask_losses = deque('')
    backdoor_losses = deque('')
    maintask_acc = deque('')
    backdoor_acc = deque('')

    if hlpr.params.track_training:
        tr_maintask_losses = deque('')
        tr_backdoor_losses = deque('')
        tr_maintask_acc = deque('')
        tr_backdoor_acc = deque('')

    distances_to_initial_solution = deque('')
    weight_distances = deque('')
    initial_model_weights = deque('')
    l1_norms = deque('')

    forward_passes_per_iteration = 0
    backward_passes_per_iteration = 0

    weights_ascending = np.linspace(0, 1, num=hlpr.params.max_continuation_iterations)
    for iteration, weight_a in enumerate(weights_ascending):
        print(f"Starting iteration {iteration} with weights {weight_a} and {1-weight_a}")
        model = SimpleNet(100).to(device="cuda")
        hlpr.task.model = model
        if iteration == 0:
            initial_model_weights = [params.clone() for params in model.parameters()]

        if hlpr.params.optimizer == "SGD":
            optimizer = torch.optim.SGD(hlpr.task.model.parameters(), lr=hlpr.params.lr, momentum=0.9)
        else:
            optimizer = torch.optim.Adam(hlpr.task.model.parameters(), lr=hlpr.params.lr)

        start_time = time.time()
        previous_model_weights = [p.clone() for p in model.parameters()]

        # Train
        weights = [float(weight_a), float(1-weight_a)]
        for epoch in range(hlpr.params.epochs):
            train_scalarized(hlpr, epoch, model, optimizer, hlpr.task.train_loader, attack=True, weights=weights)
        
        print(f"Finished training iteration {iteration} at {time.strftime('%H:%M:%S', time.gmtime(time.time() - start_time))}")
        if iteration == 0:
            forward_passes_per_iteration = model.forward_passes
            backward_passes_per_iteration = model.backward_passes
            model.reset_passes()

        next_model_weights = [params.clone() for params in model.parameters()]
        collect_loss_and_accuracy(hlpr, maintask_losses, maintask_acc, backdoor_losses, backdoor_acc, backwards=False)
        if hlpr.params.track_training:    
            collect_loss_and_accuracy(hlpr, tr_maintask_losses, tr_maintask_acc, tr_backdoor_losses, tr_backdoor_acc,
                                    training=True)
        collect_regularization(next_model_weights, l1_norms)
        collect_distance(initial_model_weights, next_model_weights, distances_to_initial_solution, backwards=False)
        collect_distance(previous_model_weights, next_model_weights, weight_distances, backwards=False)

        # 6. Intermediary plot
        if iteration % hlpr.params.plot_continuation_on_iteration == 0 \
                or iteration == hlpr.params.max_continuation_iterations - 1:
            print(f"Plotting {iteration} under {model_path}")
            plot_results(hlpr, hlpr.params.lr, hlpr.params.lr,
                        maintask_losses, backdoor_losses,
                        maintask_losses, backdoor_losses,
                        maintask_acc, backdoor_acc,
                        weight_distances,
                        distances_to_initial_solution,
                        l1_norms)

        if iteration % hlpr.params.continuation_checkpoints_at == 0 \
                or iteration == hlpr.params.max_continuation_iterations - 1:
            print(f"Checkpointin model {iteration} under {model_path}")
            hlpr.save_model(model, iteration)

        # 7. Write down all results
        print(f"Forward passes per iteration: {forward_passes_per_iteration}")
        print(f"Backward passes per iteration step: {backward_passes_per_iteration}")

    save_continuation_results(helper,
                            maintask_losses,
                            maintask_acc,
                            backdoor_losses,
                            backdoor_acc,
                            maintask_losses,
                            maintask_acc,
                            backdoor_losses,
                            backdoor_acc,
                            distances_to_initial_solution,
                            weight_distances,
                            l1_norms)
    if hlpr.params.track_training:
        save_training_continuation_results(helper,
                            tr_maintask_losses,
                            tr_maintask_acc,
                            tr_backdoor_losses,
                            tr_backdoor_acc,
                            tr_maintask_losses,
                            tr_maintask_acc,
                            tr_backdoor_losses,
                            tr_backdoor_acc,)


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
        swipe_parameters(lrs=[0.005, 0.001, 0.0005, 0.0001])
        exit()

    helper = Helper(params)
    continuation_helper = ContinuationHelper(params)
    logger.warning(create_table(params))

    try:
        if helper.params.fl:
            fl_run(helper)
        else:
            if helper.params.scalarization:
                run_scalarization(helper)
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
