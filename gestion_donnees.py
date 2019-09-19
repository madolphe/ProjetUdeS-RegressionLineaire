# -*- coding: utf-8 -*-

#####
# JOUFFROY Emma - 19 157 145
# ADOLPHE Maxime - 19 156 789
###

import numpy as np
import matplotlib.pyplot as plt


class GestionDonnees:
    """
    Classe permettant de générer des données aléatoires suivant soit un modèle
    linéaire soit un sinus, soit une tangeante hyperbolique. La première fonction
    permet de générer ces données en données de test / données cible et données
    d'entraînement / cible entrainement. La deuxième fonction permet d'afficher
    ces données avec matplotlib selon le modèle que suivent les donnée.

    """
    def __init__(self, w, modele_gen, nb_train, nb_test, bruit):
        self.w = w
        self.modele_gen = modele_gen
        self.nb_train = nb_train
        self.nb_test = nb_test
        self.bruit = bruit

    def generer_donnees(self):
        """
        Fonction qui genere des donnees de test et d'entrainement.

        modele_gen : 'lineaire', 'sin' ou 'tanh'
        nb_train : nb de donnees d'entrainement
        nb_test : nb de donnees de test
        bruit : amplitude du bruit (superieur ou egale a zero)
        """
        np.random.seed(self.nb_train)
        x_train = np.random.rand(self.nb_train)
        x_test = np.random.rand(self.nb_test)
        if self.modele_gen == 'lineaire':
            t_train = self.w[0] + x_train * self.w[1] + np.random.randn(self.nb_train) * self.bruit
            t_test = self.w[0] + x_test * self.w[1] + np.random.randn(self.nb_test) * self.bruit
        elif self.modele_gen == 'sin':
            t_train = np.sin(x_train * self.w[1] * 2) + np.random.randn(self.nb_train) * self.bruit
            t_test = np.sin(x_test * self.w[1] * 2) + np.random.randn(self.nb_test) * self.bruit
        else:
            t_train = np.tanh((x_train - 0.5) * self.w[1] * 2) + np.random.randn(self.nb_train) * self.bruit
            t_test = np.tanh((x_test - 0.5) * self.w[1] * 2) + np.random.randn(self.nb_test) * self.bruit

        return x_train, t_train, x_test, t_test

    def afficher_donnees_et_modele(self, x, t, scatter=True):
        """
        afficher des donnees

        x : vecteur de donnees
        t : vecteur de cibles
        scatter : variable determinant si on doit afficher une courbe ou des points
        """
        x_mod = np.arange(0, 1, 0.01)

        if self.modele_gen == 'lineaire':
            t_mod = self.w[0] + x_mod * self.w[1]
        elif self.modele_gen == 'sin':
            t_mod = np.sin(x_mod * self.w[1] * 2)
        else:
            t_mod = np.tanh((x_mod - 0.5) * self.w[1] * 2)

        if scatter is True:
            plt.scatter(x, t)
        else:
            idx = np.argsort(x)
            plt.plot(x[idx], t[idx], 'g')

        plt.plot(x_mod, t_mod, 'k')
        plt.ylim(ymin=-4.5, ymax=4.5)


if __name__ == '__main__':

    def test_donnees_generees():
        # exemple = sklearn:1 type_model_gen:sin nb_train:20 nb_test:20 bruit:0.3 10 0.001
        w = [0.3, 4.1]  # Parametres du modele generatif
        modele_gen = "lineaire"
        nb_train = 500
        nb_test = 20
        bruit = 0.1
        gestionnaire_donnees = GestionDonnees(w, modele_gen, nb_train, nb_test, bruit)
        [x_train, t_train, x_test, t_test] = gestionnaire_donnees.generer_donnees()
        print(x_train.shape)
        print(t_train.shape)
        plt.scatter(x_train, t_train)
        plt.show()
    # test_donnees_generees()
