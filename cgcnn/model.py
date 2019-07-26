from __future__ import print_function, division

import torch
import torch.nn as nn


class ConvLayer(nn.Module):
    """
    Convolutional operation on graphs
    """
    def __init__(self, atom_fea_len, nbr_fea_len):
        """
        Initialize ConvLayer.

        Parameters
        ----------

        atom_fea_len: int
          Number of atom hidden features.
        nbr_fea_len: int
          Number of bond features.
        """
        super(ConvLayer, self).__init__()
        self.atom_fea_len = atom_fea_len
        self.nbr_fea_len = nbr_fea_len
        # (2*64 + 41) → (2 * 64) のFC層
        self.fc_full = nn.Linear(2*self.atom_fea_len+self.nbr_fea_len,
                                 2*self.atom_fea_len)
        self.sigmoid = nn.Sigmoid()
        self.softplus1 = nn.Softplus()
        self.bn1 = nn.BatchNorm1d(2*self.atom_fea_len)
        self.bn2 = nn.BatchNorm1d(self.atom_fea_len)
        self.softplus2 = nn.Softplus()

    def forward(self, atom_in_fea, nbr_fea, nbr_fea_idx):
        """
        Forward pass

        N: Total number of atoms in the batch
        M: Max number of neighbors

        Parameters
        ----------

        atom_in_fea: Variable(torch.Tensor) shape (N, atom_fea_len)
          Atom hidden features before convolution
        nbr_fea: Variable(torch.Tensor) shape (N, M, nbr_fea_len)
          Bond features of each atom's M neighbors
        nbr_fea_idx: torch.LongTensor shape (N, M)
          Indices of M neighbors of each atom

        Returns
        -------

        atom_out_fea: nn.Variable shape (N, atom_fea_len)
          Atom hidden features after convolution

        """
        # TODO will there be problems with the index zero padding?
        # Nは、Node数
        # Mは、近接原子数=12 (Edge数: 大きさは固定, 12より少ない場合は0埋めされている)
        N, M = nbr_fea_idx.shape
        # convolution
        # 近接原子のベクトルを抜き出す -> N × M × 64
        # 64：Nodeベクトルの次元
        atom_nbr_fea = atom_in_fea[nbr_fea_idx, :]
        assert atom_nbr_fea.shape == (N, M, 64)
        # 各次元の確認
        assert atom_in_fea.shape == (N, 64)
        # この挙動は...? (この後にextendで(N, M, 64)に変換)
        assert atom_in_fea.unsqueeze(1).shape == (N, 1, 64)
        # 式(5)のzベクトルを作成する処理
        # atom_in_fea.unsqueeze(1).expand(N, M, self.atom_fea_len) : 着目している原子に関するベクトル 64次元
        # atom_nbr_fea : 近接原子に関するベクトル(各近接原子によって異なる) 64次元
        # nbr_fea : 距離に関するベクトル 41次元
        total_nbr_fea = torch.cat(
            [atom_in_fea.unsqueeze(1).expand(N, M, self.atom_fea_len),
             atom_nbr_fea, nbr_fea], dim=2)

        # 最終的な各原子が持つベクトルの次元は169次元
        assert total_nbr_fea.shape == (N, M, (2*64+41))
        # FC層 169 -> 128
        total_gated_fea = self.fc_full(total_nbr_fea)
        # batch norm
        # なんで次元落としてからbatch norm...?
        total_gated_fea = self.bn1(total_gated_fea.view(
            -1, self.atom_fea_len*2)).view(N, M, self.atom_fea_len*2)
        assert total_gated_fea.shape == (N, M, 2*self.atom_fea_len)
        
        # 128次元を仲良く分割
        # なんで...?
        nbr_filter, nbr_core = total_gated_fea.chunk(2, dim=2)
        assert nbr_filter.shape == (N, M, 64)
        assert nbr_core.shape == (N, M, 64)

        # activationに通す
        nbr_filter = self.sigmoid(nbr_filter)
        nbr_core = self.softplus1(nbr_core)

        # activationした後の各隣接原子のベクトルを足す (readout)
        nbr_sumed = torch.sum(nbr_filter * nbr_core, dim=1)
        assert nbr_sumed.shape == (N, 64)
        # batch norm 
        nbr_sumed = self.bn2(nbr_sumed)
        # 前回の情報に今回得られたnbr_sumedを足して更新ベクトルを得る (次のatom_in_feaになる)
        out = self.softplus2(atom_in_fea + nbr_sumed)
        assert out.shape == (N, 64)
        return out


# pytorchのnn.Moduleをextendしたクラスはforward処理が必要
# forwardを書けばbackwardは勝手に計算してくれる
class CrystalGraphConvNet(nn.Module):
    """
    Create a crystal graph convolutional neural network for predicting total
    material properties.
    """
    def __init__(self, orig_atom_fea_len, nbr_fea_len,
                 atom_fea_len=64, n_conv=3, h_fea_len=128, n_h=1,
                 classification=False):
        """
        Initialize CrystalGraphConvNet.

        Parameters
        ----------

        orig_atom_fea_len: int
          Number of atom features in the input.
        nbr_fea_len: int
          Number of bond features.
        atom_fea_len: int
          Number of hidden atom features in the convolutional layers
        n_conv: int
          Number of convolutional layers
        h_fea_len: int
          Number of hidden features after pooling
        n_h: int
          Number of hidden layers after pooling
        """
        super(CrystalGraphConvNet, self).__init__()
        self.classification = classification
        # 92 → 64 (default)
        self.embedding = nn.Linear(orig_atom_fea_len, atom_fea_len)
        # convolutionのlayer作成 (3層)
        self.convs = nn.ModuleList([ConvLayer(atom_fea_len=atom_fea_len,
                                    nbr_fea_len=nbr_fea_len)
                                    for _ in range(n_conv)])
        # FC層 (64 → 128)
        self.conv_to_fc = nn.Linear(atom_fea_len, h_fea_len)
        # softplusの活性化関数
        self.conv_to_fc_softplus = nn.Softplus()

        # defaultでは n_h = 1
        if n_h > 1:
            self.fcs = nn.ModuleList([nn.Linear(h_fea_len, h_fea_len)
                                      for _ in range(n_h-1)])
            self.softpluses = nn.ModuleList([nn.Softplus()
                                             for _ in range(n_h-1)])
        if self.classification:
            self.fc_out = nn.Linear(h_fea_len, 2)
        else:
            self.fc_out = nn.Linear(h_fea_len, 1)
        if self.classification:
            # nn.NLLLoss()を使っているからLogのSoftmaxを使う
            self.logsoftmax = nn.LogSoftmax(dim=1)
            self.dropout = nn.Dropout()

    def forward(self, atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx):
        """
        Forward pass

        N: Total number of atoms in the batch
        M: Max number of neighbors (12)
        N0: Total number of crystals in the batch

        Parameters
        ----------

        atom_fea: Variable(torch.Tensor) shape (N, orig_atom_fea_len)
          Atom features from atom type
        nbr_fea: Variable(torch.Tensor) shape (N, M, nbr_fea_len)
          Bond features of each atom's M neighbors
        nbr_fea_idx: torch.LongTensor shape (N, M)
          Indices of M neighbors of each atom
        crystal_atom_idx: list of torch.LongTensor of length N0
          Mapping from the crystal idx to atom idx

        Returns
        -------

        prediction: nn.Variable shape (N, )
          Atom hidden features after convolution

        """

        # 92 → 64 (default)
        atom_fea = self.embedding(atom_fea)

        # convolution層
        for conv_func in self.convs:
            atom_fea = conv_func(atom_fea, nbr_fea, nbr_fea_idx)
        
        # pooling (後で定義してある, ここでcrystal_atom_idxを使う)
        # cif_id ごとの特徴ベクトルの抽出に成功
        crys_fea = self.pooling(atom_fea, crystal_atom_idx)
        # softplus
        crys_fea = self.conv_to_fc_softplus(crys_fea)
        # FC
        crys_fea = self.conv_to_fc(crys_fea)
        # softplus
        crys_fea = self.conv_to_fc_softplus(crys_fea)

        if self.classification:
            crys_fea = self.dropout(crys_fea)

        # 1層以上のFC層を要求する場合 (default: 1層)
        if hasattr(self, 'fcs') and hasattr(self, 'softpluses'):
            for fc, softplus in zip(self.fcs, self.softpluses):
                # activationは毎回softplus...?
                crys_fea = softplus(fc(crys_fea))

        # 出力
        out = self.fc_out(crys_fea)

        # 分類の場合は最後にsoftmaxに通す
        if self.classification:
            out = self.logsoftmax(out)

        return out

    def pooling(self, atom_fea, crystal_atom_idx):
        """
        Pooling the atom features to crystal features

        N: Total number of atoms in the batch
        N0: Total number of crystals in the batch

        Parameters
        ----------

        atom_fea: Variable(torch.Tensor) shape (N, atom_fea_len)
          Atom feature vectors of the batch
        crystal_atom_idx: list of torch.LongTensor of length N0
          Mapping from the crystal idx to atom idx
        """
        assert sum([len(idx_map) for idx_map in crystal_atom_idx]) ==\
            atom_fea.data.shape[0]

        # Average Poolingを採用している
        # N × 64 -> batch_size × 64
        summed_fea = [torch.mean(atom_fea[idx_map], dim=0, keepdim=True)
                      for idx_map in crystal_atom_idx]
        
        return torch.cat(summed_fea, dim=0)
