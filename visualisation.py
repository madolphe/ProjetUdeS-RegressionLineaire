# -*- coding: utf-8 -*-

import numpy as np
import sys
import solution_regression as sr
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import gestion_donnees as gd

if __name__ == '__main__':
    w = [0.3, 4.1]  # Parametres du modele generatif
    modele_gen = "tanh"
    nb_train = 200
    nb_test = 20
    bruit = 0.1
    lamb = 0
    m = 3
    skl = False
    gestionnaire_donnees = gd.GestionDonnees(w, modele_gen, nb_train, nb_test, bruit)
    [x_train, t_train, x_test, t_test] = gestionnaire_donnees.generer_donnees()
    regression = sr.Regression(lamb, m)
    regression.entrainement(x_train, t_train, using_sklearn=skl)
    t_train = np.expand_dims(t_train, axis=1)
    print(t_train.reshape((200,)).shape)
    phi_x = regression.fonction_base_polynomiale(x_train)
    projected_x_train = np.concatenate((phi_x, t_train), axis=1)
    projected_x_pred = projected_x_train[:, :2].dot(regression.w)
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.scatter3D(projected_x_train[:, 0], projected_x_train[:, 2], projected_x_train[:, 1])
    ax.scatter3D(projected_x_train[:, 0], projected_x_pred, projected_x_train[:, 1])
    fig2 = plt.figure()
    ax2 = plt.axes()
    ax2.scatter(projected_x_train[:, 0], projected_x_train[:, 2])
    ax2.scatter(projected_x_train[:, 0], projected_x_pred)
    plt.show()
