import random
import torch
import torch.nn as nn

import utils.mol_conv as mc
from utils import trainer
from utils import mol_collate
from utils.mol_props import dim_atomic_feat

from model import KROVEX
from configs.config import SET_SEED, DATASET_NAME, DATASET_PATH, BATCH_SIZE, MAX_EPOCHS, K

def main():
    SET_SEED()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    if DATASET_NAME == 'freesolv':
        print('DATASET_NAME: ', DATASET_NAME)
        dataset = mc.read_dataset_freesolv(DATASET_PATH + '.csv')
        num_descriptors = 50
        descriptors = mol_collate.descriptor_selection_freesolv

    elif DATASET_NAME == 'esol':
        print('DATASET_NAME: ', DATASET_NAME)
        dataset = mc.read_dataset_esol(DATASET_PATH + '.csv')
        num_descriptors = 63
        descriptors = mol_collate.descriptor_selection_esol

    elif DATASET_NAME == 'scgas':
        print('DATASET_NAME: ', DATASET_NAME)
        global BATCH_SIZE
        BATCH_SIZE = 128
        dataset = mc.read_dataset_scgas(DATASET_PATH + '.csv')
        num_descriptors = 23
        descriptors = mol_collate.descriptor_selection_scgas

    random.shuffle(dataset)

    model_KROVEX = KROVEX.Net(dim_atomic_feat, 1, num_descriptors).to(device)

    # loss function
    criterion = nn.L1Loss(reduction='sum')
    # criterion = nn.MSELoss(reduction='sum')

    test_losses = dict()

    print('--------- kronecker-product with descriptor selection ---------')
    test_losses['KROVEX'] = trainer.cross_validation(dataset, model_KROVEX, criterion, K, BATCH_SIZE, MAX_EPOCHS, trainer.train_model, trainer.test_model, descriptors)
    print('test loss (KROVEX): ' + str(test_losses['KROVEX']))

    print('test_losse:', test_losses)
    print(DATASET_NAME)

if __name__ == '__main__':
    main()