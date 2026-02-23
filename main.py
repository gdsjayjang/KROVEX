import yaml

from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn

from utils.utils import SET_SEED, select_loss
from configs.registry import get_dataset_spec, build_collate_fn, bs
from utils.mol_props import dim_atomic_feat
from utils import evaluation
from configs.args import get_parser

def parse_args():
    parser = get_parser()
    p, _ = parser.parse_known_args()

    if getattr(p, 'config', None):
        with open(p.config, 'r') as f:
            default_arg = yaml.safe_load(f)
        
        valid_keys = set(vars(p).keys())
        
        for k in default_arg.keys():
            if k not in valid_keys:
                raise ValueError(f'WRONG ARG in YAML: {k} (not defined in parser)')

        parser.set_defaults(**default_arg)
    
    return parser.parse_args()
    

def main():
    args = parse_args()
    SET_SEED(args)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    # DATA LOAD
    spec = get_dataset_spec(args.dataset)
    ckpt_path = spec.ckpt_path # saved model

    dataset = spec.reader(args.dataset_path + args.dataset + '.csv')
    train_dataset, test_dataset = train_test_split(dataset, test_size = 0.2, random_state = args.seed)

    if not bs("--batch-size"):
        if spec.default_batch_size is not None:
            args.batch_size = spec.default_batch_size

    num_desc = spec.num_desc
    collate_fn = build_collate_fn(args.dataset)

    # DEFINE THE MODEL
    from model import KROVEX
    KROVEX = KROVEX.Net(dim_atomic_feat, 1, num_desc).to(device)

    # LOSS FUNC
    criterion = select_loss(args.loss)

    test_losses = dict()
    print(f'{args.backbone}, {args.dataset}, {criterion}, BATCH_SIZE:{args.batch_size}, SEED:{args.seed}')

    # KROVEX
    test_losses['KROVEX'] = evaluation.evaluation(train_dataset, test_dataset, KROVEX, criterion, args.batch_size, args.epochs, collate_fn, args.dataset, args.phase, args.save_model, ckpt_path)
    print(f'Test loss: ' + str(test_losses['KROVEX']))

    print(f'{args.backbone}, {args.dataset}, {criterion}, BATCH_SIZE:{args.batch_size}, SEED:{args.seed}')


if __name__ == '__main__':
    main()