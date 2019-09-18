# -*- coding: utf-8 -*-
import numpy as np
import sys
import solution_regression as sr
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import gestion_donnees as gd
# @TODO passage d'arguments pour le lancer depuis l'exterieur

if __name__ == '__main__':
    w = [0.3, 4.1]  # Parametres du modele generatif
    modele_gen = "tanh"
    nb_train = 100
    nb_test = 20
    bruit = 0.1
    # Parametre de la regression:
    lamb = 0
    m = 3
    skl = False
    gestionnaire_donnees = gd.GestionDonnees(w, modele_gen, nb_train, nb_test, bruit)
    [x_train, t_train, _, _] = gestionnaire_donnees.generer_donnees()
    print("x_train.shape", x_train.shape)
    print("x_train: \n", x_train)
    print("t_train.shape", t_train.shape)
    # print("t_train: \n", t_train)

    # On trouve les paramètres de regression du modèle:
    regression = sr.Regression(lamb, m)
    regression.entrainement(x_train, t_train, using_sklearn=skl)
    print("regression.w.shape:", regression.w.shape)
    print("regression.w:", regression.w)

    # phi_x est un vecteur de taille nbre_donnees x m
    # m=3 donc on crée des vecteurs de R^3
    phi_x = regression.fonction_base_polynomiale(x_train, using_sklearn=skl)
    print("phi_x.shape", phi_x.shape)
    # print("phi_x: \n", phi_x)
    # print("phi_x sans la première colonne: \n", phi_x[:, 1:])

    # On prédit avec notre modèle un vecteur de cible sur notre jeu d'apprentissage
    t_pred = phi_x.dot(regression.w)
    print("t_pred.shape: ", t_pred.shape)
    # print("t_pred: \n ", t_pred)

    # On plot nos résultats:
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    # Données dans R^3:
    ax.scatter3D(phi_x[:, 1], phi_x[:, 2], t_train, c='red')
    # ax.scatter3D(phi_x[:, 1], phi_x[:, 2], t_pred)

    # Notre modèle:
    x = np.expand_dims(np.arange(0, 1, 0.1), axis=1)
    y = x**2
    X, Y = np.meshgrid(x, y)
    ones = np.ones((10, 10))
    Z = regression.w[0]*ones + regression.w[1]*X + regression.w[2]*Y
    ax.set_title('Projection dans un espace à 3 dimension')
    ax.set_xlabel('X')
    ax.set_ylabel('X^2')
    ax.set_zlabel('Prediction (combinaison linéaire de X et X^2)')
    ax.plot_surface(X, Y, Z, linewidth=0, cmap='viridis', antialiased=False)


    fig2 = plt.figure()
    ax2 = plt.axes()
    ax2.scatter(phi_x[:, 1], t_train)
    ax2.scatter(phi_x[:, 1], t_pred)
    plt.show(block=True)
