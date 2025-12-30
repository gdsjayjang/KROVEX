import random
import torch
import torch.nn as nn

from utils import trainer
from utils import mol_collate
from utils.mol_props import dim_atomic_feat

from model import KROVEX
from configs.config import SET_SEED, BACKBONE, SPLIT, DATASET_NAME, DATASET_PATH, BATCH_SIZE, MAX_EPOCHS, K, SEED

def main():
    import utils.mol_conv as mc

    SET_SEED()
    global BATCH_SIZE

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
        BATCH_SIZE = 128
        dataset = mc.read_dataset_scgas(DATASET_PATH + '.csv')
        num_descriptors = 23
        descriptors = mol_collate.descriptor_selection_scgas

    elif DATASET_NAME == 'solubility':
        print('DATASET_NAME: ', DATASET_NAME)
        BATCH_SIZE = 256
        dataset = mc.read_dataset_solubility(DATASET_PATH + '.csv')
        num_descriptors = 30
        descriptors = mol_collate.descriptor_selection_solubility

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
    print(f'{BACKBONE}, {SPLIT}-split, {DATASET_NAME}, {criterion}, BATCH_SIZE:{BATCH_SIZE}, SEED:{SEED}')


def main_sf():
    import utils.mol_conv_scaffold as mc
    SET_SEED()
    global BATCH_SIZE

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    if DATASET_NAME == 'freesolv':
        print('DATASET_NAME: ', DATASET_NAME)
        dataset, smiles_list = mc.read_dataset_freesolv(DATASET_PATH + '.csv')
        num_descriptors = 50
        descriptors = mol_collate.descriptor_selection_freesolv

    elif DATASET_NAME == 'esol':
        BATCH_SIZE = 33
        print('DATASET_NAME: ', DATASET_NAME)
        dataset, smiles_list  = mc.read_dataset_esol(DATASET_PATH + '.csv')
        num_descriptors = 63
        descriptors = mol_collate.descriptor_selection_esol

    elif DATASET_NAME == 'scgas':
        print('DATASET_NAME: ', DATASET_NAME)
        BATCH_SIZE = 128
        dataset, smiles_list  = mc.read_dataset_scgas(DATASET_PATH + '.csv')
        num_descriptors = 23
        descriptors = mol_collate.descriptor_selection_scgas

    elif DATASET_NAME == 'solubility':
        print('DATASET_NAME: ', DATASET_NAME)
        BATCH_SIZE = 256
        dataset, smiles_list  = mc.read_dataset_solubility(DATASET_PATH + '.csv')
        num_descriptors = 30
        descriptors = mol_collate.descriptor_selection_solubility

    folds = mc.scaffold_kfold_split(smiles_list, K)

    model_KROVEX = KROVEX.Net(dim_atomic_feat, 1, num_descriptors).to(device)

    # loss function
    criterion = nn.L1Loss(reduction='sum')
    # criterion = nn.MSELoss(reduction='sum')

    test_losses = dict()

    print('--------- kronecker-product with descriptor selection ---------')
    test_losses['KROVEX'] = trainer.cross_validation_sf(dataset, model_KROVEX, criterion, folds, K, BATCH_SIZE, MAX_EPOCHS, trainer.train_model, trainer.test_model, descriptors)
    print('test loss (KROVEX): ' + str(test_losses['KROVEX']))

    print('test_losse:', test_losses)
    print(f'{BACKBONE}, {SPLIT}-split, {DATASET_NAME}, {criterion}, BATCH_SIZE:{BATCH_SIZE}, SEED:{SEED}')


if __name__ == '__main__':
    if SPLIT == 'random': main()
    elif SPLIT == 'scaffold': main_sf()