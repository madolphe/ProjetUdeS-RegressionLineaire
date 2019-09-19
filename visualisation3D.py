# -*- coding: utf-8 -*-
#####
# JOUFFROY Emma - 19 157 145
# ADOLPHE Maxime - 19 156 789
###
import numpy as np
import sys
import solution_regression as sr
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import gestion_donnees as gd

################################
# Execution en tant que script
#
# tapper python3 visualisation3D.py sin 200 0.3 0.001
#
# dans un terminal
################################

def main():
    if len(sys.argv) < 4:
        print("Usage: python visualisation3D.py modele_gen nb_train bruit lambda\n")
        print("\t modele_gen=lineaire, sin ou tanh")
        print("\t nb_train: nombre de donnees d'entrainement")
        print("\t bruit: amplitude du bruit appliqué aux données")
        print("\t lambda: lambda utilisé par le modele de Ridge\n")
        print(" exemple: python3 visualisation3D.py sin 200 0.3 0.001\n")
        return

    w = [0.3, 4.1]  # Parametres du modele generatif
    modele_gen = sys.argv[1]
    nb_train = int(sys.argv[2])
    nb_test = 0
    bruit = float(sys.argv[3])

    # Parametre de la regression:
    lamb = float(sys.argv[4])
    m = 3
    skl = False

    # Modèle génératif:
    gestionnaire_donnees = gd.GestionDonnees(w, modele_gen, nb_train, nb_test, bruit)
    [x_train, t_train, _, _] = gestionnaire_donnees.generer_donnees()

    # On trouve les paramètres de regression du modèle:
    regression = sr.Regression(lamb, m)
    regression.entrainement(x_train, t_train, using_sklearn=skl)
    print("Modèle déterminé: ", regression.w)

    # phi_x est un vecteur de taille nbre_donnees x m
    # m=3 donc on crée des vecteurs de R^3
    phi_x = regression.fonction_base_polynomiale(x_train, using_sklearn=skl)

    # On prédit avec notre modèle un vecteur de cible sur notre jeu d'apprentissage
    t_pred = phi_x.dot(regression.w)

    # On trace nos résultats:
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    # Données dans R^3:
    ax.scatter3D(phi_x[:, 1], phi_x[:, 2], t_train, c='red')

    # Notre modèle:
    x = np.expand_dims(np.arange(0, 1, 0.1), axis=1)
    y = x ** 2
    X, Y = np.meshgrid(x, y)
    ones = np.ones((10, 10))
    Z = regression.w[0] * ones + regression.w[1] * X + regression.w[2] * Y
    ax.set_title('Projection dans un espace à 3 dimensions')
    ax.set_xlabel('X')
    ax.set_ylabel('X^2')
    ax.set_zlabel('Prédiction (combinaison linéaire de X et X^2)')
    # On trace notre plan:
    ax.plot_surface(X, Y, Z, linewidth=0, cmap='viridis', antialiased=False)

    # On regarde l'efficacité du modèle sur nos données à 1 dimension:
    fig2 = plt.figure()
    ax2 = plt.axes()
    ax.set_xlabel('X')
    ax.set_ylabel('Cible')
    ax2.scatter(phi_x[:, 1], t_train, label="Cible")
    ax2.scatter(phi_x[:, 1], t_pred, label="Prediction")
    ax2.legend()
    plt.show(block=True)


if __name__ == '__main__':
    main()