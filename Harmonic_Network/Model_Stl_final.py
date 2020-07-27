import argparse
import os
import json
import numpy as np
from tqdm import tqdm
import torch
from torch.optim import SGD
import torch.utils.data
import torchvision.transforms as T
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn.functional as F
import torchnet as tnt
from torchnet.engine import Engine
import sys
sys.path.insert(0, '../')
from torch.backends import cudnn
cudnn.benchmark = True
import utils
from nested_dict import nested_dict

# Model options
parser = argparse.ArgumentParser(description='Wide Residual Networks')
parser.add_argument('--model', default='model_resnet', type=str)
parser.add_argument('--depth', default=16, type=int)
parser.add_argument('--width', default=8, type=float)
parser.add_argument('--dropout', default=0.3, type=float)
parser.add_argument('--level', default=None, type=int)
parser.add_argument('--dataset', default='STL10', type=str)
parser.add_argument('--dataroot', default='data/stl/', type=str)
parser.add_argument('--fold', default=-1, type=int)
parser.add_argument('--dtype', default='float', type=str)
parser.add_argument('--groups', default=1, type=int)
parser.add_argument('--nthread', default=1, type=int)
parser.add_argument('--seed', default=1, type=int)

# Training options
parser.add_argument('--batch_size', default=32, type=int)
parser.add_argument('--lr', default=0.1, type=float)
parser.add_argument('--epochs', default=1000, type=int, metavar='N')
parser.add_argument('--weight_decay', default=0.0005, type=float)
parser.add_argument('--nesterov', action='store_true', default=False)
parser.add_argument('--epoch_step', default='[300,400,600,800]', type=str,
                    help='json list with epochs to drop lr on')
parser.add_argument('--lr_decay_ratio', default=0.2, type=float)
parser.add_argument('--resume', default='', type=str)
parser.add_argument('--note', default='', type=str)

# Device options
parser.add_argument('--cuda', action='store_true')
parser.add_argument('--save', default='model&log', type=str, help='save model and logs in this folder')
parser.add_argument('--ngpu', default=1, type=int, help='no of GPUs to use for training')
parser.add_argument('--gpu_id', default='0', type=str, help='CUDA_VISIBLE_DEVICES ids')


def cast(params, dtype='float'):
    if isinstance(params, dict):
        return {i: cast(j, dtype) for i,j in params.items()}
    else:
        return getattr(params.cuda() if torch.cuda.is_available() else params, dtype)()
        
def flatten(params):
    return {'.'.join(i): j for i, j in nested_dict(params).items_flat() if j is not None}
        
def tensor_dict_print(params):
    kmax = max(len(key) for key in params.keys())
    for i, (key, v) in enumerate(params.items()):
        print(str(i).ljust(5), key.ljust(kmax + 3), str(tuple(v.shape)).ljust(23), torch.typename(v), v.requires_grad)
        
def download_data(opt, train):
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(np.array([125.3, 123.0, 113.9]) / 255.0,
                    np.array([63.0, 62.1, 66.7]) / 255.0)
    ])
    if train:
        transform = T.Compose([
            T.Pad(12, padding_mode='reflect'),
            T.RandomHorizontalFlip(),
            T.RandomCrop(96),
            transform
        ])
    return datasets.STL10(opt.dataroot, split="train" if train else "test", download=True, transform=transform)

def model_resnet(depth, width, num_classes, dropout, level=None):
    assert (depth - 4) % 6 == 0, 'depth should be 6n+4'
    assert level is None or level in [2, 3], 'level should be 2, 3 or None'
    n = (depth - 4) // 6
    widths = [int(v * width) for v in (16, 32, 64)]

    def gen_harmonic_params(ni, no, k, normalize=False, level=None, linear=False):
        nf = k**2 if level is None else level * (level+1) // 2
        paramdict = {'conv': utils.dct_params(ni, no, nf) if linear else utils.conv_params(ni*nf, no, 1)}
        if normalize and not linear:
            paramdict.update({'bn': utils.bnparams(ni*nf, affine=False)})
        return paramdict

    def gen_block_params(ni, no):
        return {
            'harmonic0': gen_harmonic_params(ni, no, k=3, normalize=False, level=level, linear=True),
            'harmonic1': gen_harmonic_params(no, no, k=3, normalize=False, level=level, linear=True),
            'bn0': utils.bnparams(ni),
            'bn1': utils.bnparams(no),
            'convdim': utils.conv_params(ni, no, 1) if ni != no else None,
        }

    def gen_group_params(ni, no, count):
        return {'block%d' % i: gen_block_params(ni if i == 0 else no, no)
                for i in range(count)}

    flat_params = cast(flatten({
        'dct0': utils.dct_filters(n=3, groups=3),
        'dct': utils.dct_filters(n=3, groups=int(width)*64, expand_dim=0, level=level),
        'harmonic0': gen_harmonic_params(3, 16, k=3, normalize=True, level=None),
        'group0': gen_group_params(16, widths[0], n),
        'group1': gen_group_params(widths[0], widths[1], n),
        'group2': gen_group_params(widths[1], widths[2], n),
        'bn': utils.bnparams(widths[2]),
        'fc': utils.linear_params(widths[2], num_classes),
    }))

    utils.set_requires_grad_except_bn_(flat_params)

    def harmonic_block(x, params, base, mode, stride=1, padding=1):
        y = F.conv2d(x, params['dct0'], stride=stride, padding=padding, groups=x.size(1))
        if base + '.bn.running_mean' in params:
            y = utils.batch_norm(y, params, base + '.bn', mode, affine=False)
        z = F.conv2d(y, params[base + '.conv'], padding=0)
        return z

    def lin_harmonic_block(x, params, base, mode, stride=1, padding=1):
        filt = torch.sum(params[base + '.conv'] * params['dct'][:x.size(1), ...], dim=2)
        y = F.conv2d(x, filt, stride=stride, padding=padding)
        return y

    def block(x, params, base, mode, stride):
        o1 = F.relu(utils.batch_norm(x, params, base + '.bn0', mode), inplace=True)
        y = lin_harmonic_block(o1, params, base + '.harmonic0', mode, stride=stride, padding=1)
        o2 = F.relu(utils.batch_norm(y, params, base + '.bn1', mode), inplace=True)
        if dropout > 0:
            o2 = F.dropout(o2, p=dropout, training=mode, inplace=False)
        z = lin_harmonic_block(o2, params, base + '.harmonic1', mode, stride=1, padding=1)
        if base + '.convdim' in params:
            return z + F.conv2d(o1, params[base + '.convdim'], stride=stride)
        else:
            return z + x

    def group(o, params, base, mode, stride):
        for i in range(n):
            o = block(o, params, '%s.block%d' % (base,i), mode, stride if i == 0 else 1)
        return o

    def f(input, params, mode):
        x = harmonic_block(input, params, 'harmonic0', mode, stride=2, padding=1)
        g0 = group(x, params, 'group0', mode, 1)
        g1 = group(g0, params, 'group1', mode, 2)
        g2 = group(g1, params, 'group2', mode, 2)
        o = F.relu(utils.batch_norm(g2, params, 'bn', mode))
        o = F.avg_pool2d(o, 12, 1, 0)
        o = o.view(o.size(0), -1)
        o = F.linear(o, params['fc.weight'], params['fc.bias'])
        return o

    return f, flat_params
    
def main():
    opt = parser.parse_args()
    print('parsed options:', vars(opt))
    epoch_step = json.loads(opt.epoch_step)
    log_step = 5
    if opt.fold >= 0 and opt.fold <= 9:
        log_step *= 5
        epoch_step = [ep*5 for ep in epoch_step]
        opt.epochs *= 5
 
    num_classes = 10

    torch.manual_seed(opt.seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_id

    def create_iterator(mode):
        if opt.fold < 0 or opt.fold > 9:
            return DataLoader(download_data(opt, mode), opt.batch_size, shuffle=mode,
                              num_workers=opt.nthread, pin_memory=torch.cuda.is_available())
        if mode:
            folds = np.loadtxt('fold_indices.txt', dtype=np.int64)
            fold = folds[opt.fold]
            fold = torch.from_numpy(fold)
        return DataLoader(download_data(opt, mode), opt.batch_size, sampler=SubsetRandomSampler(fold) if mode else None,
                          num_workers=opt.nthread, pin_memory=torch.cuda.is_available())

    train_loader = create_iterator(True)
    test_loader = create_iterator(False)

    kwargs = {}
    if not opt.level is None:
        kwargs.update({'level': opt.level})
    f, params = model_resnet(opt.depth, opt.width, num_classes, opt.dropout, **kwargs)
    def create_optimizer(opt, lr):
        print('creating optimizer with lr = ', lr)
        return SGD([v for v in params.values() if v.requires_grad], lr, momentum=0.9, weight_decay=opt.weight_decay, nesterov=opt.nesterov)

    optimizer = create_optimizer(opt, opt.lr)

    epoch = 0
    if opt.resume != '':
        state_dict = torch.load(opt.resume)
        epoch = state_dict['epoch']
        params_tensors = state_dict['params']
        for k, v in params.items():
            if k in params_tensors:
                v.data.copy_(params_tensors[k])
        optimizer.load_state_dict(state_dict['optimizer'])

    print('\nParameters:')
    tensor_dict_print(params)

    n_parameters = sum(p.numel() for p in params.values() if p.requires_grad)
    print('\nTotal number of parameters:', n_parameters)

    meter_loss = tnt.meter.AverageValueMeter()
    classacc = tnt.meter.ClassErrorMeter(accuracy=True)
    timer_train = tnt.meter.TimeMeter('s')
    timer_test = tnt.meter.TimeMeter('s')

    if not os.path.exists(opt.save):
        os.mkdir(opt.save)

    def h(sample):
        inputs = cast(sample[0], opt.dtype)
        targets = cast(sample[1], 'long')
        y = utils.data_parallel(f, inputs, params, sample[2], list(range(opt.ngpu))).float()
        return F.cross_entropy(y, targets), y

    def log(t, state):
        torch.save(dict(params={k: v for k, v in params.items() if k.find('dct') == -1}, epoch=t['epoch'], 
                   optimizer=state['optimizer'].state_dict()), os.path.join(opt.save, 'model.pt7'))
        z = vars(opt).copy()
        z.update(t)
        with open(os.path.join(opt.save, 'log.txt'), 'a') as flog:
            flog.write('json_stats: ' + json.dumps(z) + '\n')
        print(z)

    def on_sample(state):
        state['sample'].append(state['train'])

    def on_forward(state):
        loss = float(state['loss'])
        classacc.add(state['output'].data, state['sample'][1])
        meter_loss.add(loss)
        if state['train']:
            state['iterator'].set_postfix(loss=loss)

    def on_start(state):
        state['epoch'] = epoch

    def on_start_epoch(state):
        classacc.reset()
        meter_loss.reset()
        timer_train.reset()
        state['iterator'] = tqdm(train_loader, dynamic_ncols=True)
        epoch = state['epoch'] + 1
        if epoch in epoch_step:
            lr = state['optimizer'].param_groups[0]['lr']
            state['optimizer'] = create_optimizer(opt, lr * opt.lr_decay_ratio)

    def on_end_epoch(state):
        if state['epoch'] % log_step == 0:
            train_loss = meter_loss.value()
            train_acc = classacc.value()
            train_time = timer_train.value()
            meter_loss.reset()
            classacc.reset()
            timer_test.reset()

            with torch.no_grad():
                engine.test(h, test_loader)

            test_acc = classacc.value()[0]
            print(log({
                "train_loss": train_loss[0],
                "train_acc": train_acc[0],
                "test_loss": meter_loss.value()[0],
                "test_acc": test_acc,
                "epoch": state['epoch'],
                "train_time": train_time,
                "test_time": timer_test.value(),
            }, state))
            print('==> id: %s (%d/%d), test_acc: \33[91m%.2f\033[0m' %
                  (opt.save, state['epoch'], opt.epochs, test_acc))

    engine = Engine()
    engine.hooks['on_sample'] = on_sample
    engine.hooks['on_forward'] = on_forward
    engine.hooks['on_start_epoch'] = on_start_epoch
    engine.hooks['on_end_epoch'] = on_end_epoch
    engine.hooks['on_start'] = on_start
    engine.train(h, train_loader, opt.epochs, optimizer)


if __name__ == '__main__':
    main()