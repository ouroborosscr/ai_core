import argparse
import shutil
from datetime import datetime

import yaml
from prompt_toolkit import prompt
from tqdm import tqdm

# noinspection PyUnresolvedReferences
from dataset.pipa import Annotations  # legacy to correctly load dataset.
from helper import Helper
from utils.utils import *

def train(hlpr: Helper, epoch, model, optimizer, train_loader, attack=True):
    criterion = hlpr.task.criterion
    model.train()


    for i, data in tqdm(enumerate(train_loader)):

        batch = hlpr.task.get_batch(i, data)
        model.zero_grad()
        loss = hlpr.attack.compute_blind_loss(model, criterion, batch, attack)
        loss.backward()
        optimizer.step()

        hlpr.report_training_losses_scales(i, epoch)
        if i == hlpr.params.max_batch_id:
            break

    return

def test(hlpr: Helper, epoch, backdoor=False):
    model = hlpr.task.model
    model.eval()
    hlpr.task.reset_metrics()


    with torch.no_grad():
        for i, data in tqdm(enumerate(hlpr.task.test_loader)):

            batch = hlpr.task.get_batch(i, data)
            # if backdoor:
            #     batch = hlpr.attack.synthesizer.make_backdoor_batch(batch,
            #                                                         test=True,
            #                                                         attack=True)

            outputs = model(batch.inputs)
            hlpr.task.accumulate_metrics(outputs=outputs, labels=batch.labels)
    metric = hlpr.task.report_metrics(epoch,
                             prefix=f'Backdoor {str(backdoor):5s}. Epoch: ',
                             tb_writer=hlpr.tb_writer,
                             tb_prefix=f'Test_backdoor_{str(backdoor):5s}')

    return metric

def run(hlpr):
    acc = test(hlpr, 0, backdoor=False) #测试模型成功率
    for epoch in range(hlpr.params.start_epoch,
                       hlpr.params.epochs + 1):
        ##delete
        # if(epoch==2):
        #     break
        ##
        train(hlpr, epoch, hlpr.task.model, hlpr.task.optimizer,
              hlpr.task.train_loader)
        acc = test(hlpr, epoch, backdoor=False)
        # test(hlpr, epoch, backdoor=True)
        hlpr.save_model(hlpr.task.model, epoch, acc)

def main(paramspath,name):#paramspath参数存储路径,
    #parser = argparse.ArgumentParser(description='Ai')
    with open(paramspath) as f:
        params = yaml.load(f, Loader=yaml.FullLoader)#导入参数
    params['current_time'] = datetime.now().strftime('%b.%d_%H.%M.%S')
    params['name'] = name
    helper = Helper(params)
    try:
        if helper.params.fl:
            fl_run(helper)
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




if __name__ == '__main__':
    # parser = argparse.ArgumentParser(description='Backdoors')
    # parser.add_argument('--params', dest='params', default='utils/params.yaml')
    # parser.add_argument('--name', dest='name', required=True)

    # print(parser)

    # args = parser.parse_args()

    # print(args)

    # with open(args.params) as f:
    #     #print(f)
    #     params = yaml.load(f, Loader=yaml.FullLoader)

    # params['current_time'] = datetime.now().strftime('%b.%d_%H.%M.%S')
    # #params['commit'] = args.commit
    # params['name'] = args.name

    # print(params)

    main("configs/mnist_params.yaml","mnist")

    #helper = Helper(params)
    # logger.warning(create_table(params))

    # try:
    #     if helper.params.fl:
    #         fl_run(helper)
    #     else:
    #         run(helper)
    # except (KeyboardInterrupt):
    #     if helper.params.log:
    #         answer = prompt('\nDelete the repo? (y/n): ')
    #         if answer in ['Y', 'y', 'yes']:
    #             logger.error(f"Fine. Deleted: {helper.params.folder_path}")
    #             shutil.rmtree(helper.params.folder_path)
    #             if helper.params.tb:
    #                 shutil.rmtree(f'runs/{args.name}')
    #         else:
    #             logger.error(f"Aborted training. "
    #                          f"Results: {helper.params.folder_path}. "
    #                          f"TB graph: {args.name}")
    #     else:
    #         logger.error(f"Aborted training. No output generated.")