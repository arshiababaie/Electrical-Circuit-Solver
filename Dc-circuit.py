import pandas as pd
import numpy as np



class Circuit:
    
    '''
    A Circuit object is defined from an input data file of 4 columns (space-separated values) 
    with the following format:\n
    Component Name***From Node****To Node****Value.\n
    However in the input data file there should be no header to label the columns.\n
    Electrical components are limited to independent voltage sources, independent current sources and resistors.\n 
    The first character of the component names has the following restrictions:\n
    - For a voltage source, the name should start with a V\n
    - For a current source, the name should start with an I\n
    - For a resistor, the name should start with a R\n
    There is no restriction on node labels except for the ground wnich should be labelled with the integer 0.\n
    The units of the Value colum are:\n
    - Volt for voltage sources\n
    - Ampere for current sources\n
    - Ohm for resistors\n
    Circuits are solved using Modified Nodal Analysis (MNA).
    '''
    
    def __init__(self, inputlist_file: str):
        self.inputlist_df = pd.read_csv(inputlist_file, sep=' ', names=["Component Name", "From Node", "To Node", "Value"])
        self.inputlist_df.loc[:, ['From Node', 'To Node']] = self.inputlist_df.loc[:, ['From Node', 'To Node']].astype(str)
        self._unique_user_nodes = self._unique_nodes()
        self._nodes_transf = self._transform_nodes()
        self._inputlist_arr = self._df_to_array()
        self.no_nodes = int(self._inputlist_arr[:, [1, 2]].max())
        self.no_branches = self.inputlist_df.shape[0]
        self._inc_mat = self._calc_inc_mat()
        self._y_n = self._calc_y_n()
        self._vadj = self._inc_mat[:, self._inputlist_arr[:,0] == 10]
        self._mna_mat = self._calc_mna_mat()
        self._i_n = self._calc_i_n()
        self._v_s = self._inputlist_arr[self._inputlist_arr[:, 0] == 10, :][:, 3]
        self._rhs = np.concatenate((self._i_n, self._v_s))
        self._x = np.linalg.solve(self._mna_mat, self._rhs)
        self._v_n, self._i_v_s = self._x[:self.no_nodes], self._x[self.no_nodes:]
        self.v_b = self._inc_mat.T @ self._v_n
        self._y_b = self._calc_y_b()
        self.i_b = self._calc_i_b()
        self.branch_quatitles = self._branch_resvlts_df()
        self.node_voltages = self._v_n_user()
        
        
    def _unique_nodes(self):
        user_nodes_flat = pd.concat([self.inputlist_df['From Node'], self.inputlist_df['To Node']])
        return user_nodes_flat.unique()
    
    
    def _transform_nodes(self):
        nodes_transf = {'0':0}
        k = 1
        for user_node in self._unique_user_nodes:
            if user_node != '0':
                nodes_transf[user_node] = k
                k = k+1
        return nodes_transf
    
    
    def _df_to_array(self):
        inputlist_transf = self.inputlist_df.copy()
        inputlist_transf.replace({'From Node': self._nodes_transf, 'To Node':self._nodes_transf}, inplace=True)
        compo_types = {"R":0, "V":10, "I":20}
        inputlist_transf["Component Name"] = self.inputlist_df["Component Name"].map(lambda name: compo_types[name[0]])
        return np.array(inputlist_transf, dtype=float)
    
    def _calc_y_n(self):
        y_n = np.zeros((self.no_nodes, self.no_nodes))
        res_branch = self._inputlist_arr[self._inputlist_arr[:, 0] ==0, :]
        for idx in range (res_branch.shape[0]):
            g = 1 / res_branch[idx, 3]
            node_from = int(res_branch[idx, 1])
            node_to = int(res_branch[idx, 2])
            if node_from != 0:
                y_n[node_from -1, node_from -1] +=g
                if node_to != 0:
                    y_n[node_from - 1, node_to - 1] += -g
                    y_n[node_to - 1, node_from - 1] += -g
                    y_n[node_to - 1, node_to - 1] += g
            else:
                y_n[node_to - 1, node_to - 1] += g
        return y_n
    
    
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
    
    def _calc_mna_mat(self):
        upper_mat = np.concatenate((self._y_n, self._vadj), axis=1)
        bottom_right_mat = np.zeros((self._vadj.shape[1], self._vadj.shape[1]))
        bottom_mat = np.concatenate((self._vadj.T, bottom_right_mat), axis=1)
        return np.concatenate((upper_mat, bottom_mat), axis=0)
    
    
    def _calc_i_n(self):
        current_src_branch = self._inputlist_arr[self._inputlist_arr[:, 0] == 20, :]
        i_n = np.zeros(self.no_nodes)
        for idx in range(current_src_branch.shape[0]):
            node_from = int(current_src_branch[idx, 1])
            node_to = int(current_src_branch[idx, 2])
            if node_to != 0:
                i_n[node_from -1] += -current_src_branch[idx, 3]
            if node_to != 0:
                i_n[node_to - 1] += current_src_branch[idx, 3]
        return i_n
            
    
    def _calc_y_b(self):
        y_b = np.zeros((self.no_branches, self.no_branches))
        for k in range(self.no_branches):
            if self._inputlist_arr[k, 0] == 0:
                y_b[k,k] = 1/self._inputlist_arr[k, 3]
        return y_b
    
    def _calc_i_b(self):
        i_b = np.zeros(self.no_branches)
        current_src_mask = self._inputlist_arr[:,0] == 20
        i_b[current_src_mask] = self._inputlist_arr[current_src_mask][:,3]
        voltage_src_mask = self._inputlist_arr[:,0] == 10
        i_b[voltage_src_mask] = self._i_v_s
        i_b += self._y_b @ self.v_b
        return i_b
    
    def _branch_resvlts_df(self):
        final_df = self.inputlist_df.copy()
        final_df['Voltage (V)'] = self.v_b
        final_df['Current (A)'] = self.i_b
        return final_df
    
    
    def _v_n_user(self):
        node_labels = self._unique_user_nodes[self._unique_user_nodes != '0']
        node_voltages_list = [self._v_n[self._nodes_transf[node_labels[i]]-1] for i in range(self.no_nodes)]
        return pd.DataFrame({'Node':node_labels, 'Voltage (v)':node_voltages_list})