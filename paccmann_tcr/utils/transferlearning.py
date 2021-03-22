""""Utils for model finetuning"""
import logging

from paccmann_predictor.models import BimodalMCA

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def update_mca_model(model: BimodalMCA, params: dict) -> BimodalMCA:
    """
    Receives a pretrained model (instance of BimodalMCA), modifies it and returns
    the updated object
    Args:
        model (BimodalMCA): Pretrained model to be modified.
        params (dict): Hyperparameter file for the modifications. Needs to include:
            - number_of_tunable_layers (how many layers should not be frozen. If
                number exceeds number of existing layers, all layers are tuned.)
    Returns:
        BimodalMCA: Modified model for finetune
    """

    if not isinstance(model, BimodalMCA):
        raise TypeError(
            f'Wrong model type, was {type(model)}, not BimodalMCA.'
        )

    num_to_tune = params['number_of_tunable_layers']

    # Not strictly speaking all layers, but all param matrices, gradient-req or not.
    num_layers = len(['' for p in model.parameters()])

    # Bound tunable layers to network size
    if num_to_tune > num_layers:
        logger.warning(
            f'Model has {num_layers} tunable layers. Given # is larger: {num_to_tune}.'
            'All layers will be trained.'
        )
        num_to_tune = num_layers

    layer_names = [n for n, p in model.named_parameters()]

    # Infer how many layers exist per type
    num_dense_layers = num_layers - min(
        [
            i for i, (n, _) in enumerate(model.named_parameters())
            if 'dense_layers' in n
        ]
    )
    if 'batch_norm.weight' in layer_names:
        num_dense_layers += 2
    num_ligand_layers = len(
        [n for n, _ in model.named_parameters() if 'ligand' in n]
    )
    num_receptor_layers = num_layers - num_dense_layers - num_ligand_layers

    assert num_receptor_layers == len(
        [n for n, _ in model.named_parameters() if 'receptor' in n]
    ), 'Layer types cant be inferred uniquely.'

    logger.info(
        f'Model has {num_layers} layers. {num_dense_layers} are dense layers, '
        f'{num_ligand_layers} are ligand layers, {num_receptor_layers} for receptors.'
    )

    # Infer how many layers should be tuned
    tuned_dense = min(num_to_tune, num_dense_layers)
    tuned_receptor = min(
        max(num_to_tune - num_dense_layers, 0), num_receptor_layers
    )
    tuned_ligand = max(num_to_tune - num_dense_layers - num_receptor_layers, 0)
    logger.info(
        f'{num_to_tune} will be finetuned. {tuned_dense} of those will be dense, '
        f'{tuned_receptor} will be receptor and {tuned_ligand} will be ligand.'
    )
    # Freeze the right layers
    counter_dense, counter_ligands, counter_receptor = 0, 0, 0
    free_dense, free_ligands, free_receptor = 0, 0, 0
    for idx, (name, param) in enumerate(model.named_parameters()):
        if 'ligand' in name:
            if num_ligand_layers - tuned_ligand > counter_ligands:
                param.requires_grad = False
            else:
                free_ligands += 1
            counter_ligands += 1
        elif 'receptor' in name:
            if num_receptor_layers - tuned_receptor > counter_receptor:
                param.requires_grad = False
            else:
                free_receptor += 1
            counter_receptor += 1
        elif 'dense' or 'batch' in name:
            if num_dense_layers - tuned_dense > counter_dense:
                param.requires_grad = False
            else:
                free_dense += 1
            counter_dense += 1
        else:
            raise ValueError(f'Unknown layer type {name}.')

    assert free_dense == tuned_dense, 'Wrong number of dense layers fixed'
    assert free_ligands == tuned_ligand, 'Wrong number of Ligand layers fixed'
    assert free_receptor == tuned_receptor, 'Wrong number of Receptor layers fixed'

    for idx, (name, param) in enumerate(model.named_parameters()):
        logger.info(f'{name}, {param.shape}, Grad: {param.requires_grad}')

    return model
