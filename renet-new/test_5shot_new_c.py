import os
import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader

from common.meter import Meter
#from common.utils import compute_accuracy, load_model, setup_run, by
from common.utils import compute_accuracy, load_model, setup_run
from models.dataloader.samplers import CategoriesSampler
from models.dataloader.data_utils import dataset_builder
from models.renet import RENet


def evaluate(epoch, model, loader, args=None, set='val'):
    model.eval()

    loss_meter = Meter()
    acc_meter = Meter()

    label = torch.arange(args.way).repeat(args.query).cuda()

    k = args.way * args.shot
    tqdm_gen = tqdm.tqdm(loader)

    with torch.no_grad():
        for i, (data, labels) in enumerate(tqdm_gen, 1):
            data = data.cuda()
            model.module.mode = 'encoder'
            data = model(data)
            data_shot, data_query = data[:k], data[k:]
            model.module.mode = 'cca'

            logits = model((data_shot.unsqueeze(0).repeat(args.num_gpu, 1, 1, 1, 1), data_query))
            dists_1_t      = torch.transpose(logits, dim0=0, dim1=1)
            values,indices = torch.topk(dists_1_t,8,largest=True)
            r_f_support = data_shot
            for i in range(8):
                for j in range(5):
                    fake_support = torch.index_select(data_query,dim=0,index=indices[j][i]).view(1,640,5,5)
                    r_f_support = torch.cat((r_f_support,fake_support),dim=0)
            data_shot_new = r_f_support.contiguous().view(args.way*(args.shot+8), 640,5,5 )
            #data_shot_new = r_f_support.contiguous().view(args.way*args.shot,5, 640,5,5).mean(1).view(args.way*args.shot,640,5,5)
            logits_new = model((data_shot_new.unsqueeze(0).repeat(args.num_gpu, 1, 1, 1, 1), data_query))
            loss = F.cross_entropy(logits_new, label)
            acc = compute_accuracy(logits_new, label)

            loss_meter.update(loss.item())
            acc_meter.update(acc)
    #print('current_acc: ',acc_meter.avg(),'loss: ', loss_meter.avg())
            #tqdm_gen.set_description(f'[{set:^5}] epo:{epoch:>3} | avg.loss:{loss_meter.avg():.4f} | avg.acc:{by(acc_meter.avg())} (curr:{acc:.3f})')

    return loss_meter.avg(), acc_meter.avg(), acc_meter.confidence_interval()


def test_main(model, args):

    ''' load model '''
    model = load_model(model, os.path.join(args.save_path, 'max_acc.pth'))

    ''' define test dataset '''
    Dataset = dataset_builder(args)
    test_set = Dataset('test', args)
    sampler = CategoriesSampler(test_set.label, args.test_episode, args.way, args.shot + args.query)
    test_loader = DataLoader(test_set, batch_sampler=sampler, num_workers=8, pin_memory=True)

    ''' evaluate the model with the dataset '''
    _, test_acc, test_ci = evaluate("best", model, test_loader, args, set='test')
    print('final_test_acc',test_acc,'+-',test_ci)
    #print(f'[final] epo:{"best":>3} | {by(test_acc)} +- {test_ci:.3f}')

    return test_acc, test_ci


if __name__ == '__main__':
    args = setup_run(arg_mode='test')

    ''' define model '''
    model = RENet(args).cuda()
    model = nn.DataParallel(model, device_ids=args.device_ids)

    test_main(model, args)
