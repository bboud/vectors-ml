import numpy as np
import torch
from vectorsml.result import Result
from torch.utils.data import Dataset as ds

import h5py

class Dataset(ds):
    def __init__(self, file):
        super(Dataset, self).__init__()

        self.results = []

        data = h5py.File(file)

        def _getInitalCompTensor(result):
            for i in result.keys():
                if "eccentricities" in i: 
                    initial = result[i]
                    comp_tens = torch.transpose(torch.tensor(initial[:]), 0, 1)
                    initial_tensor = comp_tens[4] + 1j*comp_tens[5]
                    return initial_tensor
                
        def _getFinalCompTensor(result):
            final = result['particle_9999_dNdeta_pT_0.2_3.dat']
            comp_tens = torch.transpose(torch.tensor(final[:]), 0, 1)
            final_tensor = comp_tens[4] + 1j*comp_tens[5]
            return final_tensor


        for v in data.keys():
            spvn_result = data[v]

            self.results.append(
                Result(v, _getInitalCompTensor(spvn_result), _getFinalCompTensor(spvn_result))
            )

    def __len__(self):
        return len(self.results)

    def __getitem__(self, item):
        return (self.results[item].key, self.results[item].value)