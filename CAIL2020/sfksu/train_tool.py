import logging
import os
import torch
from torch.autograd import Variable
from torch.optim import lr_scheduler
from tensorboardX import SummaryWriter
import shutil
from timeit import default_timer as timer

from tqdm import tqdm

from eval_tool import valid, gen_time_str, output_value
from init_tool import init_test_dataset, init_formatter
from utils import get_csv_logger, get_path
from evaluate import p_f1

logger = logging.getLogger(__name__)


def checkpoint(filename, model, optimizer, trained_epoch, config, global_step):
    model_to_save = model.module if hasattr(model, 'module') else model
    save_params = {
        "model": model_to_save.state_dict(),
        "optimizer_name": config.get("train", "optimizer"),
        "optimizer": optimizer.state_dict(),
        "trained_epoch": trained_epoch,
        "global_step": global_step
    }

    try:
        torch.save(save_params, filename)
    except Exception as e:
        logger.warning("Cannot save models with error %s, continue anyway" % str(e))


def train(parameters, config, gpu_list, do_test=False):
    get_path("log")
    epoch_logger = get_csv_logger(
        os.path.join("log",
                     'bert-epoch.csv'),
        title='epoch,train_acc,train_f1,valid_acc,valid_f1')

    epoch = config.getint("train", "epoch")
    batch_size = config.getint("train", "batch_size")

    output_time = config.getint("output", "output_time")
    test_time = config.getint("output", "test_time")

    output_path = os.path.join(config.get("output", "model_path"), config.get("output", "model_name"))
    if os.path.exists(output_path):
        logger.warning("Output path exists, check whether need to change a name of model")
    os.makedirs(output_path, exist_ok=True)

    trained_epoch = parameters["trained_epoch"] + 1
    model = parameters["model"]
    optimizer = parameters["optimizer"]
    dataset = parameters["train_dataset"]
    global_step = parameters["global_step"]
    output_function = parameters["output_function"]

    if do_test:
        init_formatter(config, ["test"])
        test_dataset = init_test_dataset(config)

    if trained_epoch == 0:
        shutil.rmtree(
            os.path.join(config.get("output", "tensorboard_path"), config.get("output", "model_name")), True)

    os.makedirs(os.path.join(config.get("output", "tensorboard_path"), config.get("output", "model_name")),
                exist_ok=True)

    writer = SummaryWriter(os.path.join(config.get("output", "tensorboard_path"), config.get("output", "model_name")),
                           config.get("output", "model_name"))

    step_size = config.getint("train", "step_size")
    gamma = config.getfloat("train", "lr_multiplier")

    gradient_accumulation_steps = config.getint("train","gradient_accumulation_steps")
    max_grad_norm = config.getfloat("train", "max_grad_norm")

    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    exp_lr_scheduler.step(trained_epoch)

    logger.info("Training start....")

    print("Epoch  Stage  Iterations  Time Usage    Loss    Output Information")
    total_len = len(dataset)
    print('total len',total_len)
    more = ""
    if total_len < 10000:
        more = "\t"
    for epoch_num in range(trained_epoch, epoch):
        start_time = timer()
        current_epoch = epoch_num

        exp_lr_scheduler.step(current_epoch)

        acc_result = None
        total_loss = 0

        output_info = ""
        step = -1

        tqdm_obj = tqdm(dataset, ncols=80)
        for step, data in enumerate(tqdm_obj):
            for key in data.keys():
                if isinstance(data[key], torch.Tensor):
                    if len(gpu_list) > 0:
                        data[key] = Variable(data[key].cuda())
                    else:
                        data[key] = Variable(data[key])

            results = model(data, config, gpu_list, acc_result, "train")
            loss, acc_result = results["loss"], results["acc_result"]
            if gradient_accumulation_steps > 1:
                loss = loss / gradient_accumulation_steps
            total_loss += float(loss)
            loss.backward()
            # optimizer.zero_grad()

            if (step + 1) % gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

            if step % output_time == 0:
                output_info = output_function(acc_result, config)

                delta_t = timer() - start_time

                output_value(current_epoch, "train", "%d/%d" % (step + 1, total_len), "%s/%s" % (
                    gen_time_str(delta_t), gen_time_str(delta_t * (total_len - step - 1) / (step + 1))),
                             "%.3lf" % (total_loss / (step + 1)), output_info, '\r', config)

            writer.add_scalar(config.get("output", "model_name") + "_train_iter", float(loss), global_step)

        trainp_f1 = p_f1(acc_result)
        output_value(current_epoch, "train", "%d/%d" % (step + 1, total_len), "%s/%s" % (
            gen_time_str(delta_t), gen_time_str(delta_t * (total_len - step - 1) / (step + 1))),
                     "%.3lf" % (total_loss / (step + 1)), output_info, None, config)

        if step == -1:
            logger.error("There is no data given to the model in this epoch, check your data.")
            raise NotImplementedError

        checkpoint(os.path.join(output_path, "model%d.bin" % current_epoch), model, optimizer, current_epoch, config,
                   global_step)
        writer.add_scalar(config.get("output", "model_name") + "_train_epoch", float(total_loss) / (step + 1),
                          current_epoch)

        if current_epoch % test_time == 0:
            with torch.no_grad():
                validp_f1 = valid(model, parameters["valid_dataset"], current_epoch, writer, config, gpu_list, output_function)
                if do_test:
                    valid(model, test_dataset, current_epoch, writer, config, gpu_list, output_function, mode="test")

        # Logging
        l = []
        l.extend(trainp_f1)
        l.extend(validp_f1)
        l = [str(i) for i in l]
        epoch_logger.info(','.join([str(epoch_num)]+ l))
    # checkpoint(os.path.join(output_path, "%d.pkl" % current_epoch), model, optimizer, current_epoch, config,
    #            global_step)
