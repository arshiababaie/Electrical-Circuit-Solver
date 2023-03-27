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
        self.time_range = pd.read_csv(inputlist_file[:-4] + '_cond.txt',
                                           sep=' ', names=['tmin', 'tmax', 'step'])
        self.time_vec = self._get_time_vec()
        self.time_step = self.time_vec[1] - self.time_vec[0]
        self._v_s = self._get_sources('V')
        self._i_s = self._get_sources('I')
        self._masks = {compo_type: self._inputlist_arr[:, 0] == type_code
                       for compo_type, type_code in self._compo_types.items()}
        self._inc_mat = self._calc_inc_mat()
        self._vadj = self._inc_mat[:, self._masks['V']]
        self._y_b = self._calc_y_b()
        self._y_n = self._calc_y_n()
        self._mna_mat = self._calc_mna_mat()
        self.v_n = np.zeros((self.no_nodes, len(self.time_vec)))
        self.v_b = np.zeros((self.no_branches, len(self.time_vec)))
        self.i_b = np.zeros((self.no_branches, len(self.time_vec)))
        self._current_sources = self._inputlist_arr[self._masks['I']]
        self._inductors = self._inputlist_arr[self._masks['L']]
        self._capacitors = self._inputlist_arr[self._masks['C']]
        self.no_inductors = len(self._inductors)
        self.no_capacitors = len(self._capacitors)
        self.no_current_sources = len(self._current_sources)
        self._solve_circuit()
        

    def _df_to_array(self):
        inputlist_transf = self.inputlist_df.copy()
        inputlist_transf['Component Name'] = inputlist_transf['Component Name'].map(lambda name: self._compo_types[name[0]])
        return np.array(inputlist_transf, dtype=float)
    
    def _get_sources_list(self, source_nature='V'):
        sources_names = self.inputlist_df.loc[self.inputlist_df['Component Name'].str[0] == source_nature, 'Component Name']
        sources_list = []
        for source_name in sources_names:
            sources_list.append(
                pd.read_csv(os.path.join(self.dirname, source_name + '.txt'), sep=' ', names=[source_name])
            )
        return sources_list
    
    def _get_time_vec(self):
        tmin, tmax, step = self.time_range.iloc[0,:]
        return np.arange(tmin, tmax + step, step)
    
    def _get_sources(self, source_nature='V'):
        source_list = self._v_s_list if source_nature == 'V' else self._i_s_list
        sources = np.zeros((len(source_list), len(self.time_vec)))
        for k, source in enumerate(source_list):
            ramp_up = float(source.iloc[3, 0])
            mag = float(source.iloc[5, 0])
            bef_ramp = self.time_vec < ramp_up
            aft_ramp = self.time_vec >= ramp_up
            if source.iloc[1, 0] == 'DC':
                slope = mag / ramp_up
                sources[k, bef_ramp] = slope * self.time_vec[bef_ramp]
                sources[k, aft_ramp] = mag
            elif source.iloc[1, 0] == 'Sinusoidal':
                frequency = float(source.iloc[7, 0])
                w = 2 * np.pi * frequency
                slope = mag * np.sqrt(2) / ramp_up
                sources[k, bef_ramp] = slope * np.sin(w * self.time_vec[bef_ramp]) * self.time_vec[bef_ramp]
                sources[k, aft_ramp] = mag * np.sqrt(2) * np.sin(w * self.time_vec[aft_ramp])
        return sources


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
        admittance_dict = {
            self._compo_types['R']: 1 / branch[3],
            self._compo_types['L']: self.time_step / (2 * branch[3]),
            self._compo_types['C']: 2 * branch[3] / self.time_step,
        }
        return admittance_dict[branch[0]]
        
    def _calc_y_b(self):
        y_b = np.zeros((self.no_branches, self.no_branches))
        for k in range(self.no_branches):
            if self._inputlist_arr[k, 0] in [self._compo_types['R'], self._compo_types['L'], self._compo_types['C']]:
                y_b[k, k] = self._calc_branch_adm(self._inputlist_arr[k, :])
        return y_b
    
    def _calc_y_n(self):
        '''
        Yn is calculated using the algorithm defined in the MNA lectures.\n
        Note: Yn could be computed using the formula Yn = IncMat * Yb * transpose(IncMat),
        but the algorithm used in this method is more efficient as IncMat is sparse for larg circuits.
        '''
        y_n = np.zeros((self.no_nodes, self.no_nodes))
        passive_branch = self._inputlist_arr[self._masks['R'] | self._masks['L'] | self._masks['C']]
        for idx in range (passive_branch.shape[0]):
            g = self._calc_branch_adm(passive_branch[idx, :])
            node_from = int(passive_branch[idx, 1])
            node_to = int(passive_branch[idx, 2])
            if node_from != 0:
                y_n[node_from -1, node_from -1] += g
                if node_to != 0:
                    y_n[node_from - 1, node_to - 1] += -g
                    y_n[node_to - 1, node_from - 1] += -g
                    y_n[node_to - 1, node_to - 1] += g
            else:
                y_n[node_to - 1, node_to - 1] += g
        return y_n

    
    def _calc_mna_mat(self):
        upper_mat = np.concatenate((self._y_n, self._vadj), axis=1)
        bottom_right_mat = np.zeros((self._vadj.shape[1], self._vadj.shape[1]))
        bottom_mat = np.concatenate((self._vadj.T, bottom_right_mat), axis=1)
        return np.concatenate((upper_mat, bottom_mat), axis=0)
    
    
    def _calc_i_n_kth(self, ind_cur_inj_kth, cap_cur_inj_kth, kth_iter):
        i_n = np.zeros(self.no_nodes)
        branches_single_type = [self._current_sources, self._inductors, self._capacitors]
        current_injections = [self._i_s[:, kth_iter], ind_cur_inj_kth, cap_cur_inj_kth]
        for br_single_type, current_inj in zip(branches_single_type, current_injections):
            for idx in range(br_single_type.shape[0]):
                node_from = int(br_single_type[idx, 1])
                node_to = int(br_single_type[idx, 2])
                if node_from != 0:
                    i_n[node_from - 1] += -current_inj[idx]
                if node_to != 0:
                    i_n[node_to - 1] += current_inj[idx]
        return i_n
        
    
    def _calc_i_b_kth(self, ind_cur_inj_kth, cap_cur_inj_kth, i_vs_kth, kth_iter):
        i_b_kth = np.zeros(self.no_branches)
        i_b_kth[self._masks['I']] = self._i_s[:, kth_iter]
        i_b_kth[self._masks['V']] = i_vs_kth
        i_b_kth[self._masks['L']] += ind_cur_inj_kth
        i_b_kth[self._masks['C']] += cap_cur_inj_kth
        i_b_kth += self._y_b @ self.v_b[:, kth_iter]
        return i_b_kth
    
    def _calc_ind_cur_inj_kth(self, ind_vals, kth_iter):
        prev_br_cur = self.i_b[self._masks['L'], kth_iter]
        prev_res_cur = self.time_step / (2 * ind_vals) * self.v_b[self._masks['L'], kth_iter]
        return prev_br_cur + prev_res_cur
    
    
    def _calc_cap_cur_inj_kth(self, cap_vals, kth_iter):
        prev_br_cur = -self.i_b[self._masks['C'], kth_iter]
        prev_res_cur = -2 * cap_vals / self.time_step * self.v_b[self._masks['C'], kth_iter]
        return prev_br_cur + prev_res_cur
    
    def _solve_circuit(self):
        ind_cur_inj_kth = np.zeros(self.no_inductors)      # inductive current injection at the k-th time step
        cap_cur_inj_kth = np.zeros(self.no_capacitors)     # capacitive current injection at k-th time step
        ind_vals = self._inputlist_arr[self._masks['L'], 3]
        cap_vals = self._inputlist_arr[self._masks['C'], 3]
        for kth_iter in range(len(self.time_vec)):
            i_n_kth = self._calc_i_n_kth(ind_cur_inj_kth, cap_cur_inj_kth, kth_iter)
            rhs_kth = np.concatenate((i_n_kth, self._v_s[:, kth_iter]))
            x_kth = np.linalg.solve(self._mna_mat, rhs_kth)
            self.v_n[:, kth_iter], i_v_s_kth = x_kth[:self.no_nodes], x_kth[self.no_nodes:]
            self.v_b[:, kth_iter] = self._inc_mat.T @ self.v_n[:, kth_iter]
            self.i_b[:, kth_iter] = self._calc_i_b_kth(ind_cur_inj_kth, cap_cur_inj_kth, i_v_s_kth, kth_iter)
            ind_cur_inj_kth = self._calc_ind_cur_inj_kth(ind_vals, kth_iter)
            cap_cur_inj_kth = self._calc_cap_cur_inj_kth(cap_vals, kth_iter)
            
