import numpy as np  # math and array operations
import pandas as pd  # Dataset operations
import math
import scipy.linalg as la


class Module(object):
    def __init__(self, path):
        self.df_ = pd.read_excel(path)
        self.alternative = self.df_["nama_pet_shop"].tolist()
        self.criteria = self.df_.columns.tolist()[2:-2]
        self.maxmin = ["Max", "Max", "Min", "Min"]

    def eucledian(self, a, b):
        return np.sqrt(np.sum(np.square(a-b)))

    def updateDistance(self, lat, long):
        origins = np.array([lat, long])
        # Update distance
        new_df = np.array(self.df_)
        for i in range(len(new_df)):
            destination = [new_df[i][6], new_df[i][7]]
            new_df[i, 5] = self.eucledian(origins, destination)
        self.df = pd.DataFrame(new_df, columns=[
                               "id", "name", "kapasitas", "dokter", "harga", "jarak", "lat", "long"])
        self.df.pop("lat")
        self.df.pop("long")
        self.df.pop("id")
        self.df = self.df.transpose()
        self.df = self.df.drop(["name"])
        self.df.columns = self.alternative

    def paired_matrix(self, names, bobot):
        size = len(names)  # Length of the list of criteria/alternative
        # Zeros matrix to save the paired comparisons
        M = np.zeros((size, size), dtype=object)
        cont = 1  # counter to compare each pair of criteria/alternative once
        index = 0
        for i in range(size):
            for j in range(cont, size):
                # str(input("Paired comparsion between {} & {}: ".format(names[i],names[j])))
                M[i, j] = bobot[index]
                index += 1
                if "/" in M[i, j]:  # Numbers input as fraction
                    num, den = M[i, j].split("/")
                    M[i, j] = int(num)/int(den)
                    M[j, i] = int(den)/int(num)
                else:
                    M[i, j] = float(M[i, j])  # Str to float
                    M[j, i] = float(1/M[i, j])
            cont += 1
        np.fill_diagonal(M, 1)  # Fill diagonal with 1
        return M.astype(float)

    def lambda_function(self, matrix):
        vals, vects = la.eig(matrix)  # Eigenvalues and eigenvectors
        maxcol = list(vals).index(max(vals))  # Column with max eigenvalues
        # Getting the maxeigenvalues of the eigenvectors (Real part)
        eigenvect = np.real(vects[:, maxcol])
        lambda_values = eigenvect/sum(eigenvect)  # Eigenvalues normalized 0-1
        return lambda_values

    def consistency(self, matrix, lamb):
        P = matrix.sum(axis=0)  # Sum matrix columns
        lambda_max = np.dot(P, lamb)  # Dot product between P and Lambda values
        n = len(matrix)  # Number of criterias/alternatives
        CI = (lambda_max-n)/(n-1)  # Consitency index
        RI = [0, 0, 0.58, 0.89, 1.12, 1.26, 1.36, 1.41, 1.42, 1.49,
              1.52, 1.54, 1.56, 1.58]  # Random consitency index
        CR = CI/RI[n-1]  # Consitency rate
        return CR

    def weight_criteria(self, matrix, sum_column):
        jumlah_kriteria = 4
        jumlah_pet_shop = len(matrix[0])
        bobot_kriteria = []
        for row in matrix:
            bobot_kriteria.append(
                sum([row[i]/sum_column[i] for i in range(len(row))])/4)
        return bobot_kriteria

    def getBobotKriteria(self, arr_bobot_perbandingan):
        # (POST data bobot perbandingan dari User)
        bobot_perbandingan = arr_bobot_perbandingan
        condition = True  # Condition of consistency
        while condition:
            m_cri = self.paired_matrix(
                self.criteria, bobot_perbandingan)  # Paired matrix
            lambda_cri = self.lambda_function(
                m_cri)  # Max eigenvalues of criterias
            CR = self.consistency(m_cri, lambda_cri)  # Consistency test
            if CR <= 0.1:
                condition = False
            else:
                condition = True
        array_of_criteria = m_cri
        sum_column = np.zeros(len(self.criteria))  # Array to save columns sum
        for i in range(len(self.criteria)):
            sum_column[i] = np.sum(array_of_criteria[:, i])
        bobot_kriteria = self.weight_criteria(array_of_criteria, sum_column)
        df_cri = pd.DataFrame(m_cri)
        df_cri.insert(0, "Jenis Kriteria", self.criteria)
        df_cri.columns = ["Jenis Kriteria"] + self.criteria
        df_decisional = self.df.T
        df_decisional.columns = self.criteria
        # Avoid scientific notation in numpy matrix and 3 decimal digits
        np.set_printoptions(suppress=True, precision=5)
        m_decisional = df_decisional.to_numpy()
        m_rij = m_decisional**2  # Matrix to save r_ij^2  values
        sum_column = np.zeros(len(self.criteria))  # Array to save columns sum
        for i in range(len(self.criteria)):
            sum_column[i] = np.sum(m_rij[:, i])
        m_normalized = m_decisional/np.sqrt(sum_column)
        m_weight = m_normalized*bobot_kriteria
        # Arrays to save the ideal alternative
        v_max = np.zeros(len(self.criteria))
        # Arrays to save the Non-ideal alternative
        v_min = np.zeros(len(self.criteria))
        for i in range(len(self.criteria)):
            if self.maxmin[i] == "Max":
                v_max[i] = max(m_weight[:, i])
                v_min[i] = min(m_weight[:, i])
            elif self.maxmin[i] == "Min":
                v_max[i] = min(m_weight[:, i])
                v_min[i] = max(m_weight[:, i])
        # Euclidean distance to the ideal solution
        dist_max = np.zeros(len(self.alternative))
        # Euclidean distance to the Non-ideal solution
        dist_min = np.zeros(len(self.alternative))
        for i in range(len(self.alternative)):
            dist_max[i] = self.eucledian(m_weight[i, :], v_max)
            dist_min[i] = self.eucledian(m_weight[i, :], v_min)
        r_prox = dist_min/(dist_max+dist_min)
        df_end = pd.DataFrame(r_prox)
        df_end.insert(0, "Material", self.alternative)
        df_end.columns = ["Material", "Relative Proximity"]
        df_end = df_end.sort_values(by=["Relative Proximity"], ascending=False)
        df_end.insert(0, "Ranking", list(range(1, len(self.alternative)+1)))
        res = np.array(df_end)
        result = []
        for r in res:
            result.append(r[1])
        return result
