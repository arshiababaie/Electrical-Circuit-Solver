import pandas as pd
import numpy as np
import os

pi = np.pi
exp = np.exp



class Circuit:
    
    def __init__(self, inputlist_file):
        self.dirname = os.path.dirname(inputlist_file)
        #self.dirname = inputlist_file.split('/')[0] if inputlist_file.find('/') != -1 else ""
        self.inputlist_df = pd.read_csv(inputlist_file, sep=' ', names=["Component Name", "From Node", "To Node", "Value"])
        self._compo_types = {'R': 0, 'L': 1, 'C': 2, 'V':10, 'I':20}
        self._inputlist_arr = self._df_to_array()  #convert the pd data frame to np array for easier data processing
        self.no_nodes = int(self._inputlist_arr[:, [1, 2]].max())
        self.no_branches = self.inputlist_df.shape[0]
        self._v_s_list = self._get_sources_list('V')
        self._i_s_list = self._get_sources_list('I')
        self.frequency_range = pd.read_csv(inputlist_file[:-4] + '_cond.txt',
                                           sep=' ', names=['fmin', 'fmax', 'step'], dtype=float)
        self.f_vec = self._get_frequency_vec()
        self.no_frequencies = len(self.f_vec)
        self._inc_mat = self._calc_inc_mat()
        self._y_n = self._calc_y_n()
        self._vadj = self._inc_mat[:, self._inputlist_arr[:,0] == self._compo_types['V']]
        self._mna_mat = self._calc_mna_mat()
        self._v_s = self._get_sources('V')
        self._i_s = self._get_sources('I')
        self._i_n = self._calc_i_n()
        self._rhs = np.concatenate((self._i_n, self._v_s), axis=1)
        self._x = np.linalg.solve(self._mna_mat, self._rhs)
        self._v_n, self._i_v_s = self._x[:,:self.no_nodes,:], self._x[:,self.no_nodes:,:]
        self.v_b = np.einsum('ji,mjk->mik', self._inc_mat, self._v_n)        
        self._y_b = self._calc_y_b()
        self.i_b = self._calc_i_b()
        self.i_b_mag = abs(self.i_b)
        self.v_n_mag = abs(self._v_n)
        

    def _df_to_array(self):
        inputlist_transf = self.inputlist_df.copy()
        inputlist_transf['Component Name'] = inputlist_transf['Component Name'].map(lambda name: self._compo_types[name[0]])
        return np.array(inputlist_transf, dtype=float)
    
    def _get_sources_list(self, source_nature='V'):
        sources_data = self.inputlist_df.loc[self.inputlist_df['Component Name'].str[0] == source_nature, 'Component Name']
        sources_list = []
        for source_name in sources_data:
            sources_list.append(
                [
                    source_name,
                    pd.read_csv(os.path.join(self.dirname, source_name + '.txt'),
                                sep=' ', names=['Magnitude', 'Angle'])
                    ]
                )
        return sources_list
    
    def _get_frequency_vec(self):
        fmin, fmax, step = self.frequency_range.iloc[0,:]
        if step == 0.0:
            f_vec = fmin
        else:
            f_vec = [fmin + k * step for k in range(int((fmax - fmin) / step) + 1)]
        f_vec = np.array(f_vec)
        return f_vec

    def _calc_inc_mat(self):
        inc_mat = np.zeros((self.no_nodes, self.no_branches))
        for b in range(0, self.no_branches):
            node_from = int(self._inputlist_arr[b, 1])
            node_to = int(self._inputlist_arr[b, 2])            
            if node_from != 0:
                inc_mat[node_from - 1, b] = 1
            if node_to !=0:
                inc_mat[node_to-1, b] = -1
        return inc_mat    


    def _calc_branch_adm(self, branch):
        if branch[0] == self._compo_types['R']:
            return 1 / branch[3]
        elif branch[0] == self._compo_types['L']:
            return 1 / (1j * 2 * pi * self.f_vec * branch[3])
        elif branch[0] == self._compo_types['C']:
            return 1j * 2 * pi * self.f_vec * branch[3]
    
    def _calc_y_n(self):
        '''
        Yn is calculated using the algorithm defined in the MNA lectures.\n
        Note: Yn could be computed using the formula Yn = IncMat * Yb * transpose(IncMat),
        but the algorithm used in this method is more efficient as IncMat is sparse for larg circuits.
        '''
        y_n = np.zeros((self.no_frequencies, self.no_nodes, self.no_nodes), dtype=complex)
        mask_r = self._inputlist_arr[:, 0] == self._compo_types['R']
        mask_l = self._inputlist_arr[:, 0] == self._compo_types['L']
        mask_c = self._inputlist_arr[:, 0] == self._compo_types['C']
        passive_branch = self._inputlist_arr[mask_r | mask_l | mask_c]
        for idx in range (passive_branch.shape[0]):
            y = self._calc_branch_adm(passive_branch[idx, :])
            node_from = int(passive_branch[idx, 1])
            node_to = int(passive_branch[idx, 2])
            if node_from != 0:
                y_n[:, node_from -1, node_from -1] += y
                if node_to != 0:
                    y_n[:, node_from - 1, node_to - 1] += -y
                    y_n[:, node_to - 1, node_from - 1] += -y
                    y_n[:, node_to - 1, node_to - 1] += y
            else:
                y_n[:, node_to - 1, node_to - 1] += y
        return y_n

    
    def _calc_mna_mat(self):
        vadj_repeat = np.repeat(self._vadj[np.newaxis, :, :], self.no_frequencies, axis=0)
        upper_mat = np.concatenate((self._y_n, vadj_repeat), axis=2)
        bottom_right_mat = np.zeros((self.no_frequencies, self._vadj.shape[1], self._vadj.shape[1]))
        bottom_mat = np.concatenate((vadj_repeat.transpose(0,2,1), bottom_right_mat), axis=2)
        return np.concatenate((upper_mat, bottom_mat), axis=1)
    
    def _get_sources(self,source_nature='V'):
        sources_list = self._v_s_list if source_nature == 'V' else self._i_s_list
        sources = np.zeros((self.no_frequencies, len(sources_list), 1), dtype=complex)
        for source_idx in range(len(sources_list)):
            source_values_mag = sources_list[source_idx][1].iloc[:, 0]
            source_values_ang = sources_list[source_idx][1].iloc[:, 1] * pi / 180
            sources[:, source_idx, 0] = source_values_mag * exp(1j*source_values_ang)
        return sources
    
    
    def _calc_i_n(self):
        '''
        In is calculated using the algorithm defined in the MNA lectures.\n
        Note: In could be computed using the formula In = IncMat * Ib_is,
        but the algorithm used in this method is more efficient as IncMat is sparse for larg circuits.
        '''
        current_src_branch = self._inputlist_arr[self._inputlist_arr[:, 0] == self._compo_types['I'], :]
        i_n = np.zeros((self.no_frequencies, self.no_nodes, 1), dtype=complex)
        for idx in range(current_src_branch.shape[0]):
            node_from = int(current_src_branch[idx, 1])
            node_to = int(current_src_branch[idx, 2])
            if node_to != 0:
                i_n[:, node_from -1, 0] += -self._i_s[:, idx, 0]
            if node_to != 0:
                i_n[:, node_to - 1, 0] += self._i_s[:, idx, 0]
        return i_n
            
    
    def _calc_y_b(self):
        y_b = np.zeros((self.no_frequencies, self.no_branches, self.no_branches), dtype=complex)
        for k in range(self.no_branches):
            if self._inputlist_arr[k, 0] in [self._compo_types['R'], self._compo_types['L'], self._compo_types['C']]:
                y_b[:,k,k] = self._calc_branch_adm(self._inputlist_arr[k, :])
        return y_b
    
    def _calc_i_b(self):
        i_b = np.zeros((self.no_frequencies, self.no_branches, 1), dtype=complex)
        current_src_mask = self._inputlist_arr[:,0] == self._compo_types['I']
        i_b[:,current_src_mask,:] = self._i_s
        voltage_src_mask = self._inputlist_arr[:, 0] == self._compo_types['V']
        i_b[:,voltage_src_mask,:] = self._i_v_s
        i_b += self._y_b @ self.v_b
        return i_b
    
