import pandas as pd
import numpy as np
from pandas import DataFrame
from numpy import ndarray


class Model:
    """the following class gets syntaxe and the data to prepare the structure of a model"""
    def __init__(self, name: str, data):
        self.name = name
        self.data = data

    def structure(self):
        """
        prepare the syntaxe
        :return:
        all type of variables and equations
        """
        splitted = self.name.replace(" ", "")
        splitted = splitted.split("\n")
        all_vars = self.name.replace(" ", "")
        for char in ["+", "\n", "~=", "~", "~~"]:
            all_vars = all_vars.replace(char, ",")
        all_vars = list(dict.fromkeys(all_vars.split(",")))
        if '' in all_vars:
            all_vars.remove('')
        equations_dict = dict()
        for i in range(len(splitted)):
            equations_dict["equation " + str(i+1)] = splitted[i]
        variables = dict()
        latent = []
        for char in range(len(splitted)):
            if "~=" in equations_dict['equation ' + str(char+1)]:
                latent.append(equations_dict['equation ' + str(char + 1)].split("~")[0])
        variables['latent'] = latent
        variables['observed'] = [varss for varss in all_vars if varss not in latent]
        variables['endo'] = []
        for char in range(len(splitted)):
            if '~' in equations_dict["equation " + str(char+1)]:
                if ('~~'not in equations_dict["equation " + str(char+1)]) \
                        and ('~='not in equations_dict["equation " + str(char+1)]):
                    variables['endo'].append(equations_dict["equation " + str(char+1)].split("~")[0])
        variables['exo'] = [var for var in all_vars if var not in variables['endo'] + variables['observed']]
        return variables, equations_dict

    def get_beta(self):
        """
        gets output of structure methods
        :return:
        the shape of the matrix B and the parameters inside the matrix
        """
        p = len(self.structure()[0]['endo'])
        beta = np.zeros((p, p))
        for endo1 in self.structure()[0]['endo']:
            index1 = self.structure()[0]['endo'].index(endo1)
            for equation in self.structure()[1].keys():
                if "~~" not in self.structure()[1][equation]:
                    if self.structure()[1][equation].split("~")[0] == endo1:
                        for endo2 in self.structure()[0]['endo']:
                            if endo2 != endo1:
                                index2 = self.structure()[0]['endo'].index(endo2)
                                if endo2 in self.structure()[1][equation]:
                                    beta[index1, index2] = 1
        return beta

    def get_gamma(self):
        """
                gets output of structure methods
                :return:
                the shape of the matrix Gamma and the parameters inside the matrix
                """
        p = len(self.structure()[0]['endo'])
        q = len(self.structure()[0]['exo'])
        gamma = np.zeros((p, q))
        for endo in self.structure()[0]['endo']:
            index1 = self.structure()[0]['endo'].index(endo)
            for equation in self.structure()[1].keys():
                if "~~" not in self.structure()[1][equation]:
                    if self.structure()[1][equation].split("~")[0] == endo:
                        for exo in self.structure()[0]['exo']:
                            index2 = self.structure()[0]['exo'].index(exo)
                            if exo in self.structure()[1][equation]:
                                gamma[index1, index2] = 1
        return gamma

    def get_phi(self):
        """
                gets output of structure methods
                :return:
                the shape of the matrix Phi and the parameters inside the matrix
                """
        q = len(self.structure()[0]['exo'])
        phi = np.eye(q)
        for exo1 in self.structure()[0]['exo']:
            index1 = self.structure()[0]['exo'].index(exo1)
            for equation in self.structure()[1].keys():
                if self.structure()[1][equation].split("~~")[0] == exo1:
                    for exo2 in self.structure()[0]['exo']:
                        index2 = self.structure()[0]['exo'].index(exo2)
                        if exo2 in self.structure()[1][equation]:
                            phi[index1, index2] = phi[index2, index1] = 1
        return phi

    def get_psi(self):
        """
                gets output of structure methods
                :return:
                the shape of the matrix psi and the parameters inside the matrix
                """
        p = len(self.structure()[0]['endo'])
        psi = np.eye(p)
        for endo1 in self.structure()[0]['endo']:
            index1 = self.structure()[0]['endo'].index(endo1)
            for equation in self.structure()[1].keys():
                if self.structure()[1][equation].split("~~")[0] == endo1:
                    for endo2 in self.structure()[0]['endo']:
                        index2 = self.structure()[0]['endo'].index(endo2)
                        if endo2 in self.structure()[1][equation]:
                            psi[index1, index2] = psi[index2, index1] = 1
        return psi

    def get_psi_xi(self):
        """
                gets output of structure methods
                :return:
                the shape of the matrix psi_xi (covariances between exogenous variables and disturbances
                and the parameters inside the matrix
                """
        p = len(self.structure()[0]['endo'])
        q = len(self.structure()[0]['exo'])
        psi_xi = np.zeros((p, q))
        for endo in self.structure()[0]['endo']:
            index1 = self.structure()[0]['endo'].index(endo)
            for equation in self.structure()[1].keys():
                if self.structure()[1][equation].split("~~")[0] == endo:
                    for exo in self.structure()[0]['exo']:
                        index2 = self.structure()[0]['exo'].index(exo)
                        if exo in self.structure()[1][equation]:
                            psi_xi[index1, index2] = 1
        return psi_xi

    def get_lambda(self):
        """
                gets output of structure methods
                :return:
                the shape of the matrix Lamda and the parameters inside the matrix
                """
        n = len(self.structure()[0]['latent'])
        m = len(self.structure()[0]['observed'])
        lambdal = np.zeros((m, n))
        for lat in self.structure()[0]['latent']:
            index1 = self.structure()[0]['latent'].index(lat)
            for equation in self.structure()[1].keys():
                if self.structure()[1][equation].split("~=")[0] == lat:
                    for obs in self.structure()[0]['observed']:
                        index2 = self.structure()[0]['observed'].index(obs)
                        if obs in self.structure()[1][equation]:
                            lambdal[index2, index1] = 1
        return lambdal

    def get_theta(self):
        """
                gets output of structure methods
                :return:
                the shape of the matrix Theta and the parameters inside the matrix
                """
        m = len(self.structure()[0]['observed'])
        thetal = np.eye(m)
        return thetal

    def get_a_star(self):
        p = len(self.structure()[0]['endo'])
        return np.block([[self.get_gamma(), np.eye(p), self.get_beta()]])

    def get_phi_star(self):
        return np.block([[self.get_phi(), self.get_psi_xi().T],
                         [self.get_psi_xi(), self.get_psi()]])

    def load_data(self):
        """
                gets output of structure methods
                :return:
                the covariance matrix of manifest variables
                """
        if type(self.data) == DataFrame:
            data = self.data.loc[:, self.structure()[0]['observed']]
            return data.cov()
        elif type(self.data) == ndarray and self.data == self.data.T:
            eigen = np.linalg.eigh(self.data)[0]
            if sum(eigen < 0) == 0:
                return self.data
            else:
                print("problem with u're cov matrix, non semi definite!!")
                return None
        else:
            print("data should be either of type DataFrame or a numpy array")
            return None

    def get_params(self):
        """
        gets output of several methods of Model class
        :return:
        a list of model parameters initialized by start_pt
        """
        start_pt = .5
        params = []
        gamma = self.get_gamma().copy()
        beta = self.get_beta().copy()
        phi = self.get_phi().copy()
        psi = self.get_psi().copy()
        psi_xi = self.get_psi_xi().copy()
        lambdal = self.get_lambda().copy()
        thetal = self.get_theta().copy()
        p, q = gamma.shape
        m = lambdal.shape[0]
        phi_non_redo = list(phi[np.tril_indices(q)]).copy()
        psi_non_redo = list(psi[np.tril_indices(p)]).copy()
        thetal_non_redo = thetal[np.diag_indices(m)].copy()
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

    def get_matrices(self, params):
        """
        gets a lest of parameters and output of several methods of Model class
        :return:
        the matrices with updated parameters
        """
        # params = self.get_params()
        par_get = list(params.copy())
        p = self.get_gamma().shape[0]
        q = self.get_gamma().shape[1]
        m = self.get_lambda().shape[0]
        beta_new = np.zeros((p, p))
        gamma_new = np.zeros((p, q))
        phi_new = np.zeros((q, q))
        psi_new = np.zeros((p, p))
        psi_xi_new = np.zeros((p, q))
        lambdal_new = np.zeros((m, q + p))
        thetal_new = np.zeros((m, m))
        for index1 in range(p):
            for index2 in range(q):
                if self.get_gamma()[index1, index2] == 1:
                    gamma_new[index1, index2] = par_get[0]
                    par_get.pop(0)
        for index1 in range(p):
            for index2 in range(p):
                if self.get_beta()[index1, index2] == 1:
                    beta_new[index1, index2] = par_get[0]
                    par_get.pop(0)
        for index1 in range(q):
            for index2 in range(q):
                if self.get_phi()[index1, index2] == 1:
                    phi_new[index1, index2] = phi_new[index2, index1] = par_get[0]
                    par_get.pop(0)
        for index1 in range(p):
            for index2 in range(p):
                if self.get_psi()[index1, index2] == 1:
                    psi_new[index1, index2] = psi_new[index2, index1] = par_get[0]
                    par_get.pop(0)
        for index1 in range(p):
            for index2 in range(q):
                if self.get_psi_xi()[index1, index2] == 1:
                    psi_xi_new[index1, index2] = par_get[0]
                    par_get.pop(0)
        for index1 in range(m):
            for index2 in range(q + p):
                if self.get_lambda()[index1, index2] == 1:
                    lambdal_new[index1, index2] = par_get[0]
                    par_get.pop(0)
        for index1 in range(m):
            for index2 in range(m):
                if self.get_theta()[index1, index2] == 1:
                    thetal_new[index1, index2] = par_get[0]
                    par_get.pop(0)
        return gamma_new, beta_new, phi_new, psi_new, psi_xi_new, lambdal_new, thetal_new

    def compute_sigma_jor(self, params):
        """
                gets a lest of parameters and output of several methods of Model class
                :return:
                the covariance matrix implied by the model using Joreskog's formula
                """
        p, q = self.get_matrices(params)[0].shape
        gamma_j, beta_j, phi_j, psi_j, psi_xi_j, lambdal_j, thetal_j = list(self.get_matrices(params)).copy()
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
        gamma_f, beta_f, phi_f, psi_f, psi_xi_f, lambdal_f, thetal_f = list(self.get_matrices(params)).copy()
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
        p, q = self.get_matrices(params)[0].shape
        gamma_n1, beta_n1, phi_n1, psi_n1, psi_xi_n1, lambdal_n1, thetal_n1 = list(self.get_matrices(params)).copy()
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

    def compute_sigma_new_fim2(self, params):
        """
        gets a lest of parameters and output of several methods of Model class
        :return:
        the covariance matrix implied by the model using new_fim2
        """
        p, q = self.get_matrices(params)[0].shape
        gamma_n2, beta_n2, phi_n2, psi_n2, psi_xi_n2, lambdal_n2, thetal_n2 = list(self.get_matrices(params)).copy()
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

    def max_likelihood(self, params, compute_method):
        """
        :param params: a liest of model parameters
        :param compute_method: 'jor', 'fim', 'new_fim1', 'new_fim2'
        :return: the maximim likelihhod descripency function
        """
        cov_ml = self.load_data()
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

    def g_least_squares(self, params, compute_method):
        """
        :param params: a liest of model parameters
        :param compute_method: 'jor', 'fim', 'new_fim1', 'new_fim2'
        :return: the Generalized Least-Squares descripency function
        """
        cov_ml = self.load_data()
        m = self.get_matrices(params)[5].shape[0]
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

    def u_least_squares(self, params, compute_method):
        """
        :param params: a liest of model parameters
        :param compute_method: 'jor', 'fim', 'new_fim1', 'new_fim2'
        :return: the Unweighted Least-Squares descripency function
        """
        cov_ml = self.load_data()
        if compute_method == "fim":
            return np.trace((self.compute_sigma_fim(params) - cov_ml).dot((self.compute_sigma_fim(params) - cov_ml).T))
        elif compute_method == 'jor':
            return np.trace((self.compute_sigma_jor(params) - cov_ml).dot((self.compute_sigma_jor(params) - cov_ml).T))
        elif compute_method == 'new_fim1':
            return np.trace((self.compute_sigma_new_fim1(params) - cov_ml).dot(
                (self.compute_sigma_new_fim1(params) - cov_ml).T))
        elif compute_method == 'new_fim2':
            return np.trace((self.compute_sigma_new_fim2(params) - cov_ml).dot(
                (self.compute_sigma_new_fim2(params) - cov_ml).T))
        else:
            print("not a method. Try : 'fim', 'jor', 'new_fim1', or 'new_fim2'")
            return None


if __name__ == '__main__':
    df1 = pd.read_csv('data_poli.csv')
    mod = """xi_1~=x1+x2+x3
    eta_1 ~= y1+y2+y3+y4
    eta_2 ~= y5+y6+y7+y8
    eta_1~ xi_1
    eta_2~ eta_1 + xi_1"""
    my_model = Model(mod, df1)
    print(my_model.structure())
    # print(df1.cov())
    # print(my_model.load_data())
    # my_opt = opt(my_model, .5)
    # my_para = my_opt.get_params()

    # print(my_model.get_gamma())
    # print(my_model.load_data())
