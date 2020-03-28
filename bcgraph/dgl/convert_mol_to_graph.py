"""
File Name:          convert_mol_to_graph.py
Project:            biochem-graph

File Description:

"""
import logging
from typing import Optional, List, Sequence, Callable, Any, Set

from rdkit import RDLogger
from rdkit.Chem import Mol

# suppress RDKit warnings and errors
RDLogger.logger().setLevel(RDLogger.CRITICAL)

_LOGGER = logging.getLogger(__name__)



if __name__ == '__main__':

    # Example SMILES from daylight.com website
    example_smiles_list = [
        'CCc1nn(C)c2c(=O)[nH]c(nc12)c3cc(ccc3OCC)S(=O)(=O)N4CCN(C)CC4',
        'Cc1nnc2CN=C(c3ccccc3)c4cc(Cl)ccc4-n12',
        'CC(C)(N)Cc1ccccc1',
        'CN1C(=O)CN=C(c2ccccc2)c3cc(Cl)ccc13',
        'CN(C)C(=O)Cc1c(nc2ccc(C)cn12)c3ccc(C)cc3',
        'COc1ccc2[nH]c(nc2c1)S(=O)Cc3ncc(C)c(OC)c3C',
        'CS(=O)(=O)c1ccc(cc1)C2=C(C(=O)OC2)c3ccccc3',
        'Fc1ccc(cc1)C2CCNCC2COc3ccc4OCOc4c3',
        'CC(C)c1c(C(=O)Nc2ccccc2)c(c(c3ccc(F)cc3)n1CC[C@@H]4C[C@@H](O)CC('
        '=O)O4)c5ccccc5',
        'CN1CC(=O)N2[C@@H](c3[nH]c4ccccc4c3C[C@@H]2C1=O)c5ccc6OCOc6c5',
        'O=C1C[C@H]2OCC=C3CN4CC[C@@]56[C@H]4C[C@H]3[C@H]2[C@H]6N1c7ccccc75',
        'COC(=O)[C@H]1[C@@H]2CC[C@H](C[C@@H]1OC(=O)c3ccccc3)N2C',
        'COc1ccc2nccc([C@@H](O)[C@H]3C[C@@H]4CCN3C[C@@H]4C=C)c2c1',
        'CN1C[C@@H](C=C2[C@H]1Cc3c[nH]c4cccc2c34)C(=O)O',
        'CCN(CC)C(=O)[C@H]1CN(C)[C@@H]2Cc3c[nH]c4cccc(C2=C1)c34',
        'CN1CC[C@]23[C@H]4Oc5c3c(C[C@@H]1[C@@H]2C=C[C@@H]4O)ccc5O',
        'CN1CC[C@]23[C@H]4Oc5c3c(C[C@@H]1[C@@H]2C=C[C@@H]4OC(=O)C)ccc5OC(=O)C',
        'CN1CCC[C@H]1c2cccnc2',
        'Cn1cnc2n(C)c(=O)n(C)c(=O)c12',
        'C/C(=C\\CO)/C=C/C=C(/C)\\C=C\\C1=C(C)CCCC1(C)C',
    ]

    for s in example_smiles_list:

        m: Chem.Mol = Chem.MolFromSmiles(s)

        t = mol_to_tokens(m, 64)
        fp = mol_to_fingerprints(m)
        d = mol_to_descriptors(m)

        g = convert_mol_to_graph(m, True, True, 128)
        print(g)
        # assert n.shape[0] == m.GetNumAtoms()
        # assert adj.shape[1] == e.shape[0]
        # Convert edge attributes to adjacency matrix
        # tmp = torch.masked_select(adj, mask=e[:, 2].byte()).view(2, -1)

    # Test molecular similarity matrix generation
    m_list = [Chem.MolFromSmiles(s) for s in example_smiles_list]
    sim_mat = mols_to_sim_mat(m_list,
                              fp_func_list=list(FP_FUNC_DICT.keys()),
                              sim_func_list=list(SIM_FUNC_DICT.keys()))

    # Test substructure feature
    # benzene = Chem.MolFromSmiles('c1ccccc1')
    # xylene = Chem.MolFromSmiles('Cc1c(C)cccc1')
    # glucose = Chem.MolFromSmiles('OC[C@H]1OC(O)[C@H](O)[C@@H](O)[C@@H]1O')
    # sucrose = Chem.MolFromSmiles('O1[C@H](CO)[C@@H](O)[C@H](O)[C@@H](O)[C@H]'
    #                              '1O[C@@]2(O[C@@H]([C@@H](O)[C@@H]2O)CO)CO')
    # lactose = Chem.MolFromSmiles('C([C@@H]1[C@@H]([C@@H]([C@H]([C@@H](O1)O'
    #                              '[C@@H]2[C@H](O[C@H]([C@@H]([C@H]2O)O)O)CO)'
    #                              'O)O)O)O')
    #
    # print(sucrose.GetSubstructMatch(benzene))
    # print(lactose.HasSubstructMatch(glucose))
    ssm_mat = mols_to_ssm_mat(m_list)