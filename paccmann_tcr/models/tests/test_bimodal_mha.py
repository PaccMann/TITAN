"""Testing Bimodal MHA."""
import unittest

import torch

from paccmann_tcr.models import BimodalMHA


class TestMHA(unittest.TestCase):

    def test_mha(self) -> None:

        params = {
            'smiles_padding_length': 20,
            'protein_padding_length': 10,
            'peptide_embedding': 'smiles',
            'embedding_size': 4,
            'smiles_vocabulary_size': 89,
            'protein_vocabulary_size': 22,
            'num_heads': 4,
            'smiles_filters': [32, 32, 32],
            'protein_kernel_sizes': [[3, 4], [7, 4], [9, 4]],
            'smiles_kernel_sizes': [[5, 4], [7, 4], [11, 4]]
        }

        for linear in [-2, 16]:
            for conv in [True, False]:
                params.update({
                    'linear_size': linear,
                    'do_conv': conv,
                })

                model = BimodalMHA(params)

                bs = 17
                smiles = torch.randint(low=0, high=89, size=(bs, 20))
                proteins = torch.randint(low=0, high=22, size=(bs, 10))
                model.training = False
                a, b = model(smiles, proteins)
                self.assertListEqual(list(a.shape), [bs, 1])
