# -*- coding: utf-8 -*-

#####
# JOUFFROY Emma
# ADOLPHE Maxime - 79 156 789 @TODO verif le matricule et ajouter celui de Emm
###
import gestion_donnees as gd
import numpy as np
import random
from sklearn import linear_model
from math import *


class Regression:
    def __init__(self, lamb, m=1):
        self.lamb = lamb
        self.w = None
        self.M = m

    def fonction_base_polynomiale(self, x):
        """
        Fonction de base qui projette la donnee x vers un espace polynomial tel que mentionne au chapitre 3.
        Si x est un scalaire, alors phi_x sera un vecteur à self.M dimensions : (x^1,x^2,...,x^self.M)
        Si x est un vecteur de N scalaires, alors phi_x sera un tableau 2D de taille NxM

        NOTE : En mettant phi_x = x, on a une fonction de base lineaire qui fonctionne pour une regression lineaire
        """
        # AJOUTER CODE ICI
        phi_x = np.array([x**i for i in range(1, self.M+1)]).T
        return phi_x

    def recherche_hyperparametre(self, X, t, Mmin, Mmax):
        """
        Validation croisee de type "k-fold" pour k utilisee pour trouver la meilleure valeur pour
        l'hyper-parametre self.M.

        Le resultat est mis dans la variable self.M

        X: vecteur de donnees
        t: vecteur de cibles
        Mmin : plus petite valeur de M à tester
        Mmax : plus grande valeur de M à tester

        """

        #@TODO Rajouter le fait qu'on travaille uniquement sur des scalaires pour le moment

        t = np.expand_dims(t, axis=1)
        liste_m = []
        liste_erreurs = []
        taille_validation = int(np.round(X.shape[0]/10))
        for M in range(Mmin, Mmax+1):
            error_val_total = 0
            self.M = M
            phi_X = self.fonction_base_polynomiale(X)
            D = np.concatenate((phi_X, t), axis=1)
            for j in range(0, 10):
                np.random.shuffle(D)
                (X_train, t_train) = (D[taille_validation:, :-1], D[taille_validation:, D.shape[1]-1])
                (X_val, t_val) = (D[0:taille_validation, :-1], D[0:taille_validation, D.shape[1]-1])
                Wmap = np.linalg.solve((X_train.T.dot(X_train) + np.identity(X_train.shape[1])*self.lamb),
                                       (X_train.T.dot(t_train)))
                t_pred_val = X_val.dot(Wmap)
                erreur_val = self.erreur(t_val, t_pred_val)
                error_val_total = error_val_total + erreur_val
            erreur_val_moy = error_val_total / 10
            liste_m.append(M)
            liste_erreurs.append(erreur_val_moy)
        index_best_m = liste_erreurs.index(min(liste_erreurs))
        self.M = liste_m[index_best_m]
        print(self.M)
        return self.M

    def entrainement(self, X, t, using_sklearn=False):
        """
        Entraîne la regression lineaire sur l'ensemble d'entraînement forme des
        entrees ``X`` (un tableau 2D Numpy, ou la n-ieme rangee correspond à l'entree
        x_n) et des cibles ``t`` (un tableau 1D Numpy ou le
        n-ieme element correspond à la cible t_n). L'entraînement doit
        utiliser le poids de regularisation specifie par ``self.lamb``.

        Cette methode doit assigner le champs ``self.w`` au vecteur
        (tableau Numpy 1D) de taille D+1, tel que specifie à la section 3.1.4
        du livre de Bishop.
        
        Lorsque using_sklearn=True, vous devez utiliser la classe "Ridge" de 
        la librairie sklearn (voir http://scikit-learn.org/stable/modules/linear_model.html)
        
        Lorsque using_sklearn=Fasle, vous devez implementer l'equation 3.28 du
        livre de Bishop. Il est suggere que le calcul de ``self.w`` n'utilise
        pas d'inversion de matrice, mais utilise plutôt une procedure
        de resolution de systeme d'equations lineaires (voir np.linalg.solve).

        Aussi, la variable membre self.M sert à projeter les variables X vers un espace polynomiale de degre M
        (voir fonction self.fonction_base_polynomiale())

        NOTE IMPORTANTE : lorsque self.M <= 0, il faut trouver la bonne valeur de self.M

        """
        #AJOUTER CODE ICI
        if self.M <= 0:
            self.recherche_hyperparametre(X, t)

        phi_x = self.fonction_base_polynomiale(X)
        self.w = [0, 1]

    def prediction(self, x):
        """
        Retourne la prediction de la regression lineaire
        pour une entree, representee par un tableau 1D Numpy ``x``.

        Cette methode suppose que la methode ``entrainement()``
        a prealablement ete appelee. Elle doit utiliser le champs ``self.w``
        afin de calculer la prediction y(x,w) (equation 3.1 et 3.3).
        """
        # AJOUTER CODE ICI
        return 0.5

    @staticmethod
    def erreur(t, prediction):
        """
        Retourne l'erreur de la difference au carre entre
        la cible ``t`` et la prediction ``prediction``.
        On ne considère pas la régression avec prédictions multiples ( donc prediction et p sont des vecteurs à une dimension )
        """
        # AJOUTER CODE ICI
        mse = np.sum((prediction-t)**2)
        return mse


if __name__ == '__main__':

    def test_fonction_base_polynomiale():
        reg = Regression(lamb=5, m=5)
        x_int = 2
        x = np.array([1, 2, 3, 4])
        # Vérification de la fonction polynomiale appliquée:
        print(reg.fonction_base_polynomiale(x_int))
        print(reg.fonction_base_polynomiale(x))
        # Vérification de la taille de la matrice résultat:
        assert reg.fonction_base_polynomiale(x_int).shape == (reg.M,)
        assert reg.fonction_base_polynomiale(x).shape == (x.shape[0], reg.M)
    #test_fonction_base_polynomiale()

    def test_fonction_recherche_hyperparametres():
        reg = Regression(lamb=5, m=5)
        w = [0.3, 4.1]  # Parametres du modele generatif
        modele_gen = "lineaire"
        nb_train = 499
        nb_test = 20
        bruit = 0.1
        gestionnaire_donnees = gd.GestionDonnees(w, modele_gen, nb_train, nb_test, bruit)
        [x_train, t_train, x_test, t_test] = gestionnaire_donnees.generer_donnees()
        M = reg.recherche_hyperparametre(t_train, x_train, 2, 10)

    test_fonction_recherche_hyperparametres()


 # def recherche_hyperparametre(self, X, t, Mmin, Mmax):    En utilisant fenêtre glissante
 #        """
 #        Validation croisee de type "k-fold" pour k utilisee pour trouver la meilleure valeur pour
 #        l'hyper-parametre self.M.
 #
 #        Le resultat est mis dans la variable self.M
 #
 #        X: vecteur de donnees
 #        t: vecteur de cibles
 #        Mmin : plus petite valeur de M à tester
 #        Mmax : plus grande valeur de M à tester
 #
 #        On propose pour le k-fold d'utiliser une fenêtre glissante comprenant 10% du jeu d'apprentissage :
 #        Ainsi une donnee n'apparait qu'une seule fois dans le jeu de validation
 #        """
 #
 #        #@TODO Rajouter le fait qu'on travaille uniquement sur des scalaires pour le moment
 #        #self.M = 1
 #        if X.ndim == 1:
 #            pivot = int(np.round(X.shape[0]/10))
 #        elif X.ndim > 1:
 #            pivot = int(np.round(X.shape[1]/10))
 #        #else breakpoint()
 #        print("Taille X_val ( = pivot ) :", pivot)
 #        list_m_errors = []
 #
 #        for M in range(Mmin, Mmax+1):
 #            error_val_total = 0
 #            self.M = M
 #            phi_X = self.fonction_base_polynomiale(X)
 #            for j in range(0, 10):
 #                X_val = phi_X[:, j*pivot:pivot*(j+1)]
 #                X_train = np.concatenate((phi_X[:, :j*pivot], phi_X[:, pivot*(j+1):]), axis=1)
 #                t_val = t[j*pivot:pivot*(j+1)]
 #                t_train = np.concatenate((t[:j*pivot], t[pivot*(j+1):]))
 #                print("X_train.T.shape : ", X_train.T.shape)
 #                print("X_train.shape : ", X_train.shape)
 #                print("np.identity(X_train.shape[1]).shape : ", np.identity(X_train.shape[1]).shape)
 #                print("self.lamb.shape", self.lamb)
 #                print("t_train", t_train.shape)
 #                print("X_train.T.dot(t_train)", X_train.T.dot(t_train).shape)
 #
 #                # Wmap = np.linalg.solve((X_train.T.dot(X_train) + np.identity(X_train.shape[1]).dot(self.lamb)), (X_train.T.dot(t_train)))
 #                # print(Wmap.shape)
 #                # print(Wmap)
 #                #t_pred_val = X_val.T*Wmap
 #                #erreur_val = self.erreur(t_val, t_pred_val)
 #                #error_val_total = error_val_total + erreur_val
 #            #erreur_val_moy = error_val_total / 10
 #            #list_m_errors.append([M, erreur_val_moy])
 #        #index_best_m = np.where(list_m_errors == np.amin(list_m_errors))
 #       # index_best_m = np.argmin(np.array(list_m_errors[:, 1]))
 #        self.M = list_m_errors[0][0]
 #        return self.M