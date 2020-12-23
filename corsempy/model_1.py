import pandas as pd
import numpy as np
from pandas import DataFrame
from numpy.linalg import LinAlgError


class Model:
    """the following class gets syntax and the data to prepare the structure of a model"""

    def __init__(self, name: str, data: DataFrame, type_dt: str):
        self.name = name
        self.data = data
        self.type_dt = type_dt

    @property
    def structure(self):
        """
        prepare the syntaxe
        :return:
        all type of variables and equations
        """
        splitted_no_space = self.name.replace(" ", "")
        splitted_list = splitted_no_space.split("\n")
        splitted_list_no_com = splitted_list.copy()
        splitted_list_no_com = [var for var in splitted_list_no_com if (var != '' and var[0] != "#")]
        measur_eq = [var for var in splitted_list_no_com if '~=' in var]
        reg_eq = [var for var in splitted_list_no_com if ('~' in var and '~~' not in var and '~=' not in var)]
        cov_eq = [var for var in splitted_list_no_com if '~~' in var]

        # #### define type of model : path/ factor anal/ sem with lat vars
        path_anal = False
        conf_anal = False
        sem_anal = False
        if not measur_eq:
            path_anal = True
        elif not reg_eq:
            conf_anal = True
        elif measur_eq and reg_eq:
            sem_anal = True
        else:
            raise Exception('The syntax in {} is not well specified!'.format(self.name))
        var_dict = {}
        if path_anal:
            # ########## path analysis part variables
            var_path = [equation.split('~') for equation in reg_eq]
            var_path = [item for sublist in var_path for item in sublist]
            var_path = [var.split('+') for var in var_path]
            var_path = [item for sublist in var_path for item in sublist]
            observed = list(dict.fromkeys(var_path))
            endo_var = [var[:var.index("~")] for var in reg_eq]
            for var1 in endo_var:
                if endo_var.count(var1) > 1:
                    raise Exception('The variable {} should have one regression equation'.format(var1))
            exo_var = list(dict.fromkeys([var for var in var_path if var not in endo_var]))
            latent = []
            for var in var_path:
                try:
                    if var not in self.col_data:
                        raise Exception('{} is not a variable in data'.format(var))
                except AttributeError:
                    raise Exception(' data should be of type DataFrame with columns names')
            var_dict = {'latent': latent, 'observed': observed, 'endogenous': endo_var, 'exogenous': exo_var}
        elif conf_anal:
            # ###### measurement part variables
            var_measur = [equation.split('~=') for equation in measur_eq]
            latent = [var[0] for var in var_measur]
            observed_all_measur = [var[1].split('+') for var in var_measur]
            observed = [item for sublist in observed_all_measur for item in sublist]
            observed = list(dict.fromkeys(observed))
            endo_var = []
            exo_var = latent.copy()
            for var in observed:
                try:
                    if var not in self.col_data:
                        raise Exception('{} is not a variable in data'.format(var))
                except AttributeError:
                    raise Exception(' data should be of type DataFrame with columns names')
            var_dict = {'latent': latent, 'observed': observed, 'endogenous': endo_var, 'exogenous': exo_var}

        elif sem_anal:
            # ###### measurement part variables
            var_measur = [equation.split('~=') for equation in measur_eq]
            latent = [var[0] for var in var_measur]
            observed_all_measur = [var[1].split('+') for var in var_measur]
            observed = [item for sublist in observed_all_measur for item in sublist]
            observed = list(dict.fromkeys(observed))
            var_path = [equation.split('~') for equation in reg_eq]
            var_path = [item for sublist in var_path for item in sublist]
            var_path = [var.split('+') for var in var_path]
            var_path = [item for sublist in var_path for item in sublist]
            endo_var = [var[:var.index("~")] for var in reg_eq]
            for var1 in endo_var:
                if endo_var.count(var1) > 1:
                    raise Exception('The variable {} should have one regression equation'.format(var1))
            exo_var = list(dict.fromkeys([var for var in var_path if var not in endo_var]))
            var_dict = {'latent': latent, 'observed': observed, 'endogenous': endo_var, 'exogenous': exo_var}
        reg_eq_split = [var.split('~') for var in reg_eq]
        reg_eq_split_1 = [[var[0], var[1].split('+')] for var in reg_eq_split]
        measur_eq_split = [var.split('~=') for var in measur_eq]
        measur_eq_split_1 = [[var[0], var[1].split('+')] for var in measur_eq_split]
        cov_eq_split = [var.split('~~') for var in cov_eq]
        equations_model = {'reg': reg_eq_split_1, 'measure': measur_eq_split_1, 'cov': cov_eq_split}
        type_model = {'path': path_anal, 'fact': conf_anal, 'sem': sem_anal}
        return var_dict, equations_model, type_model

    @property
    def get_gamma(self):
        if self.structure[2]['fact']:
            return None
        else:
            p, q = len(self.structure[0]['endogenous']), len(self.structure[0]['exogenous'])
            gamma = np.zeros((p, q))
            for endo in self.structure[0]['endogenous']:
                index1 = self.structure[0]['endogenous'].index(endo)
                for exo in self.structure[0]['exogenous']:
                    index2 = self.structure[0]['exogenous'].index(exo)
                    for equation in self.structure[1]['reg']:
                        if equation[0] == endo and exo in equation[1]:
                            gamma[index1, index2] = 1
            return gamma

    @property
    def get_beta(self):
        if self.structure[2]['fact']:
            return None
        else:
            p = len(self.structure[0]['endogenous'])
            beta = np.zeros((p, p))
            for endo1 in self.structure[0]['endogenous']:
                index1 = self.structure[0]['endogenous'].index(endo1)
                for endo2 in self.structure[0]['endogenous']:
                    index2 = self.structure[0]['endogenous'].index(endo2)
                    for equation in self.structure[1]['reg']:
                        if equation[0] == endo1 and endo2 in equation[1]:
                            beta[index1, index2] = 1
            return beta

    @property
    def get_phi(self):
        q = len(self.structure[0]['exogenous'])
        phi = np.eye(q)
        for exo1 in self.structure[0]['exogenous']:
            index1 = self.structure[0]['exogenous'].index(exo1)
            for exo2 in self.structure[0]['exogenous']:
                index2 = self.structure[0]['exogenous'].index(exo2)
                for equation in self.structure[1]['cov']:
                    if equation[0] == exo1 and equation[1] == exo2:
                        phi[index1, index2] = phi[index2, index1] = 1
        return phi

    @property
    def get_psi(self):
        if self.structure[2]['fact']:
            return None
        else:
            p = len(self.structure[0]['endogenous'])
            psi = np.eye(p)
            for endo1 in self.structure[0]['endogenous']:
                index1 = self.structure[0]['endogenous'].index(endo1)
                for endo2 in self.structure[0]['endogenous']:
                    index2 = self.structure[0]['endogenous'].index(endo2)
                    for equation in self.structure[1]['cov']:
                        if equation[0] == endo1 and equation[1] == endo2:
                            psi[index1, index2] = psi[index2, index1] = 1
            return psi

    @property
    def get_zeta_xi(self):
        if self.structure[2]['fact']:
            return None
        else:
            p, q = len(self.structure[0]['endogenous']), len(self.structure[0]['exogenous'])
            psi_xi = np.zeros((p, q))
            for endo in self.structure[0]['endogenous']:
                index1 = self.structure[0]['endogenous'].index(endo)
                for exo in self.structure[0]['exogenous']:
                    index2 = self.structure[0]['exogenous'].index(exo)
                    for equation in self.structure[1]['cov']:
                        if endo in equation and exo in equation:
                            psi_xi[index1, index2] = 1
            return psi_xi

    @property
    def get_lambda(self):
        if self.structure[2]['path']:
            return None
        else:
            m, n = len(self.structure[0]['observed']), len(self.structure[0]['latent'])
            lambdal = np.zeros((m, n))
            for lat in self.structure[0]['latent']:
                index1 = self.structure[0]['latent'].index(lat)
                for obs in self.structure[0]['observed']:
                    index2 = self.structure[0]['observed'].index(obs)
                    for equation in self.structure[1]['measure']:
                        if equation[0] == lat and obs in equation[1]:
                            lambdal[index2, index1] = 1
            return lambdal

    @property
    def get_theta(self):
        if self.structure[2]['path']:
            return None
        else:
            m = len(self.structure[0]['observed'])
            thetal = np.eye(m)
            for obs1 in self.structure[0]['observed']:
                index1 = self.structure[0]['observed'].index(obs1)
                for obs2 in self.structure[0]['observed']:
                    index2 = self.structure[0]['observed'].index(obs2)
                    for equation in self.structure[1]['cov']:
                        if equation[0] == obs1 and equation[1] == obs2:
                            thetal[index1, index2] = thetal[index2, index1] = 1
            return thetal

    @property
    def col_data(self):
        return self.load_data.columns

    @property
    def load_data(self):
        if self.type_dt == 'data':
            try:
                if self.data.equals(self.data.T):
                    print("Warning. You be have to choose 'cov' as type of data!!!")
                    print("make sure it is not a covariance matrix!!")
                return self.data.cov()
            except AttributeError:
                raise Exception('data should be of type DataFrame with columns names')
            except ValueError:
                raise Exception('data should be of type DataFrame with columns names')
        elif self.type_dt == 'cov':
            try:
                eigen = np.linalg.eigh(self.data)[0]
                if sum(eigen < 0) == 0 and self.data.equals(self.data.T):
                    return self.data
                else:
                    raise Exception('It seems that your data is not symmetric \
                    positive semi-definite to be a covariance matrix')
            except LinAlgError:
                raise Exception('It seems that your data is not squared to be a covariance matrix.')

        else:
            raise Exception('{} is not in ("data", "cov".'.format(self.type_dt))

    def get_matrices(self, params):
        """
        gets a lest of parameters and output of several methods of Model class
        :return:
        the matrices with updated parameters
        """
        if self.structure[2]['sem']:
            # params = self.get_params()
            par_get = list(params.copy())
            p = self.get_gamma.shape[0]
            q = self.get_gamma.shape[1]
            m = self.get_lambda.shape[0]
            beta_new = np.zeros((p, p))
            gamma_new = np.zeros((p, q))
            phi_new = np.zeros((q, q))
            psi_new = np.zeros((p, p))
            psi_xi_new = np.zeros((p, q))
            lambdal_new = np.zeros((m, q + p))
            thetal_new = np.zeros((m, m))
            for index1 in range(p):
                for index2 in range(q):
                    if self.get_gamma[index1, index2] == 1:
                        gamma_new[index1, index2] = par_get[0]
                        par_get.pop(0)
            for index1 in range(p):
                for index2 in range(p):
                    if self.get_beta[index1, index2] == 1:
                        beta_new[index1, index2] = par_get[0]
                        par_get.pop(0)
            for index1 in range(q):
                for index2 in range(q):
                    if self.get_phi[index1, index2] == 1:
                        phi_new[index1, index2] = phi_new[index2, index1] = par_get[0]
                        par_get.pop(0)
            for index1 in range(p):
                for index2 in range(p):
                    if self.get_psi[index1, index2] == 1:
                        psi_new[index1, index2] = psi_new[index2, index1] = par_get[0]
                        par_get.pop(0)
            for index1 in range(p):
                for index2 in range(q):
                    if self.get_zeta_xi[index1, index2] == 1:
                        psi_xi_new[index1, index2] = par_get[0]
                        par_get.pop(0)
            for index1 in range(m):
                for index2 in range(q + p):
                    if self.get_lambda[index1, index2] == 1:
                        lambdal_new[index1, index2] = par_get[0]
                        par_get.pop(0)
            for index1 in range(m):
                for index2 in range(m):
                    if self.get_theta[index1, index2] == 1:
                        thetal_new[index1, index2] = par_get[0]
                        par_get.pop(0)
            return gamma_new, beta_new, phi_new, psi_new, psi_xi_new, lambdal_new, thetal_new
        else:
            print("not yet")

    def compute_sigma_jor(self, params):
        """
                gets a lest of parameters and output of several methods of Model class
                :return:
                the covariance matrix implied by the model using Joreskog's formula
                """
        p, q = self.get_matrices(params)[0].shape
        gamma_j, beta_j, phi_j, psi_j, psi_xi_j, lambdal_j, thetal_j = list(self.get_matrices(params))
        inv_mat = np.linalg.inv(np.eye(p) - beta_j)
        sigma1_j = phi_j
        sigma2_j = (phi_j.dot(gamma_j.T)).dot(inv_mat.T)
        sigma3_j = sigma2_j.T
        sigma4_j = (inv_mat.dot(gamma_j.dot(phi_j.dot(gamma_j.T)) + psi_j)).dot(inv_mat.T)
        return lambdal_j.dot(np.block([[sigma1_j, sigma2_j], [sigma3_j, sigma4_j]]).dot(lambdal_j.T)) + thetal_j

    def compute_sigma_fim(self, params):
        """
                        gets a lest of parameters and output of several methods of Model class
                        :return:
                        the covariance matrix implied by the model using FIM
                        """
        p, q = self.get_matrices(params)[0].shape
        gamma_f, beta_f, phi_f, psi_f, psi_xi_f, lambdal_f, thetal_f = list(self.get_matrices(params))
        a_f = np.block([[gamma_f, beta_f]])

        def recure(j):
            if j == 0:
                return phi_f
            else:
                line_f = a_f[j - 1, :q + j - 1].dot(recure(j - 1))
                line_f = line_f.reshape(1, q + j - 1)
                return np.block([[recure(j - 1), line_f.T],
                                 [line_f, 1]])

        return lambdal_f.dot(recure(p).dot(lambdal_f.T)) + thetal_f

    def compute_sigma_new_fim1(self, params):
        """
        gets a lest of parameters and output of several methods of Model class
        :return:
        the covariance matrix implied by the model using new_fim1
        """
        if self.structure[2]['sem']:
            p, q = self.get_matrices(params)[0].shape
            gamma_n1, beta_n1, phi_n1, psi_n1, psi_xi_n1, lambdal_n1, thetal_n1 = list(self.get_matrices(params))
            a_star_n1 = np.block([[gamma_n1, np.eye(p), beta_n1]])
            phi_star_n1 = np.block([[phi_n1, psi_xi_n1.T],
                                    [psi_xi_n1, psi_n1]])
            g_extract = np.block([[np.eye(q), np.zeros((q, p)), np.zeros((q, p))],
                                  [np.zeros((p, q)), np.zeros((p, p)), np.eye(p)]])

            def recure_1(j):
                if j == 0:
                    return phi_star_n1
                else:
                    line_n1 = a_star_n1[j - 1, :q + p + j - 1].dot(recure_1(j - 1))
                    line_n1 = line_n1.reshape(1, q + p + j - 1)
                    return np.block([[recure_1(j - 1), line_n1.T],
                                     [line_n1, 1]])

            return lambdal_n1.dot((g_extract.dot(recure_1(p).dot(g_extract.T))).dot(lambdal_n1.T)) + thetal_n1
        else:
            print("not yet")

    def compute_sigma_new_fim2(self, params):
        """
        gets a lest of parameters and output of several methods of Model class
        :return:
        the covariance matrix implied by the model using new_fim2
        """
        if self.structure[2]['sem']:
            p, q = self.get_matrices(params)[0].shape
            gamma_n2, beta_n2, phi_n2, psi_n2, psi_xi_n2, lambdal_n2, thetal_n2 = list(self.get_matrices(params))
            a_star_n2 = np.block([[gamma_n2, np.eye(p), beta_n2]])
            phi_star_n2 = np.block([[phi_n2, psi_xi_n2.T],
                                    [psi_xi_n2, psi_n2]])
            g_extract = np.block([[np.eye(q), np.zeros((q, p)), np.zeros((q, p))],
                                  [np.zeros((p, q)), np.zeros((p, p)), np.eye(p)]])

            def recure_2(j):
                if j == 0:
                    return phi_star_n2
                else:
                    line_n2 = a_star_n2[j - 1, :q + p + j - 1].dot(recure_2(j - 1))
                    line_n2 = line_n2.reshape(1, q + p + j - 1)
                    diag_n2 = a_star_n2[j - 1, :q + p + j - 1].dot(line_n2.T)
                    return np.block([[recure_2(j - 1), line_n2.T],
                                     [line_n2, diag_n2]])

            return lambdal_n2.dot((g_extract.dot(recure_2(p).dot(g_extract.T))).dot(lambdal_n2.T)) + thetal_n2
        else:
            print("not yet")

    def max_likelihood(self, params, compute_method):
        """
        :param params: a liest of model parameters
        :param compute_method: 'jor', 'fim', 'new_fim1', 'new_fim2'
        :return: the maximim likelihhod descripency function
        """
        if self.structure[2]['sem']:
            cov_ml = self.load_data
            if compute_method == "fim":
                return np.trace(cov_ml.dot(np.linalg.inv(self.compute_sigma_fim(params)))) + np.log(
                    np.linalg.det(self.compute_sigma_fim(params)))
            elif compute_method == 'jor':
                return np.trace(cov_ml.dot(np.linalg.inv(self.compute_sigma_jor(params)))) + np.log(
                    np.linalg.det(self.compute_sigma_jor(params)))
            elif compute_method == 'new_fim1':
                return np.trace(cov_ml.dot(np.linalg.inv(self.compute_sigma_new_fim1(params)))) + np.log(
                    np.linalg.det(self.compute_sigma_new_fim1(params)))
            elif compute_method == 'new_fim2':
                return np.trace(cov_ml.dot(np.linalg.inv(self.compute_sigma_new_fim2(params)))) + np.log(
                    np.linalg.det(self.compute_sigma_new_fim2(params)))
            else:
                print("not a method. Try : 'fim', 'jor', 'new_fim1', or 'new_fim2'")
                return None
        else:
            print("not yet")

    def g_least_squares(self, params, compute_method):
        """
        :param params: a liest of model parameters
        :param compute_method: 'jor', 'fim', 'new_fim1', 'new_fim2'
        :return: the Generalized Least-Squares descripency function
        """
        if self.structure[2]['sem']:
            cov_ml = self.load_data
            m = self.structure[0]['observed']
            if compute_method == "fim":
                return np.trace(np.eye(m) - self.compute_sigma_fim(params).dot(np.linalg.inv(cov_ml)))
            elif compute_method == 'jor':
                return np.trace(np.eye(m) - self.compute_sigma_jor(params).dot(np.linalg.inv(cov_ml)))
            elif compute_method == 'new_fim1':
                return np.trace(np.eye(m) - self.compute_sigma_new_fim1(params).dot(np.linalg.inv(cov_ml)))
            elif compute_method == 'new_fim2':
                return np.trace(np.eye(m) - self.compute_sigma_new_fim2(params).dot(np.linalg.inv(cov_ml)))
            else:
                print("not a method. Try : 'fim', 'jor', 'new_fim1', or 'new_fim2'")
                return None
        else:
            print("not yet")

    def u_least_squares(self, params, compute_method):
        """
        :param params: a liest of model parameters
        :param compute_method: 'jor', 'fim', 'new_fim1', 'new_fim2'
        :return: the Unweighted Least-Squares descripency function
        """
        if self.structure[2]['sem']:
            cov_ml = self.load_data
            if compute_method == "fim":
                return np.trace((self.compute_sigma_fim(params) - cov_ml).dot(
                    (self.compute_sigma_fim(params) - cov_ml).T))
            elif compute_method == 'jor':
                return np.trace((self.compute_sigma_jor(params) - cov_ml).dot(
                    (self.compute_sigma_jor(params) - cov_ml).T))
            elif compute_method == 'new_fim1':
                return np.trace((self.compute_sigma_new_fim1(params) - cov_ml).dot(
                    (self.compute_sigma_new_fim1(params) - cov_ml).T))
            elif compute_method == 'new_fim2':
                return np.trace((self.compute_sigma_new_fim2(params) - cov_ml).dot(
                    (self.compute_sigma_new_fim2(params) - cov_ml).T))
            else:
                print("not a method. Try : 'fim', 'jor', 'new_fim1', or 'new_fim2'")
                return None
        else:
            print("not yet")

    @property
    def get_params(self):
        """
        gets output of several methods of Model class
        :return:
        a list of model parameters initialized by start_pt
        """
        if self.structure[2]['sem']:
            start_pt = .5
            params = []
            gamma = self.get_gamma
            beta = self.get_beta
            phi = self.get_phi
            psi = self.get_psi
            psi_xi = self.get_zeta_xi
            lambdal = self.get_lambda
            thetal = self.get_theta
            p, q = gamma.shape
            m = lambdal.shape[0]
            phi_non_redo = list(phi[np.tril_indices(q)])
            psi_non_redo = list(psi[np.tril_indices(p)])
            thetal_non_redo = thetal[np.diag_indices(m)]
            for index in range(gamma.reshape(p * q).size):
                if gamma.reshape(p * q)[index] == 1:
                    params.append(start_pt)
            for index in range(beta.reshape(p ** 2).size):
                if beta.reshape(p ** 2)[index] == 1:
                    params.append(start_pt)
            for index in range(len(phi_non_redo)):
                if phi_non_redo[index] == 1:
                    params.append(start_pt)
            for index in range(len(psi_non_redo)):
                if psi_non_redo[index] == 1:
                    params.append(start_pt)
            for index in range(psi_xi.reshape(q * p).size):
                if psi_xi.reshape(q * p)[index] == 1:
                    params.append(start_pt)
            for index in range(lambdal.reshape((m * (q + p))).size):
                if lambdal.reshape(m * (q + p))[index] == 1:
                    params.append(start_pt)
            for index in range(len(thetal_non_redo)):
                if thetal_non_redo[index] == 1:
                    params.append(start_pt)
            return params
        else:
            print("not yet")


if __name__ == '__main__':
    df1 = pd.read_csv('data_poli.csv')
    df2 = 1.5
    mod_sem = """# measurement errors
    xi_1~=x1+x2+x3
    eta_1 ~= y1+y2+y3+y4
    eta_2 ~= y5+y6+y7+y8
    # regressions
    eta_1~ xi_1
    eta_2~ eta_1 + xi_1
    eta_2~~xi_1
    """
    # df2 = df1.cov()
    my_md = Model(mod_sem, df1, 'data')
    avector = my_md.get_params
    print(my_md.g_least_squares(avector, 'fim'))
