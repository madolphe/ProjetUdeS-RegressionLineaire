# -*- coding: utf-8 -*-

#####
# JOUFFROY Emma
# ADOLPHE Maxime - 79 156 789 @TODO verif le matricule et ajouter celui de Emm
###
import gestion_donnees as gd
import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures


class Regression:
    def __init__(self, lamb, m=1):
        self.lamb = lamb
        self.w = None
        self.M = m

    def fonction_base_polynomiale(self, x, using_sklearn=False):
        """
        Fonction de base qui projette la donnee x vers un espace polynomial tel que mentionne au chapitre 3.
        Si x est un scalaire, alors phi_x sera un vecteur à self.M dimensions : (x^0,x^1,x^2,...,x^self.M-1)
        Si x est un vecteur de N scalaires, alors phi_x sera un tableau 2D de taille NxM

        NOTE : En mettant phi_x = x, on a une fonction de base lineaire qui fonctionne pour une regression lineaire
        """
        if not using_sklearn:
            # Après avoir testé les résultats renvoyés par scikit learn, la fonction de base polynomiale renvoie
            # un tableau 2D de float64. C'est pour cel que nous avons décidé de caster nos données
            phi_x = np.array([x**i for i in range(0, self.M)]).T.astype(float)
        else:
            poly = PolynomialFeatures(degree=self.M - 1)
            poly_x = np.array(x).reshape(-1, 1)
            phi_x = poly.fit_transform(poly_x)
        return phi_x

    def recherche_hyperparametre(self, X, t, Mmin=1, Mmax=10):
        """
        Validation croisee de type "k-fold" pour k utilisee pour trouver la meilleure valeur pour
        l'hyper-parametre self.M.

        Le resultat est mis dans la variable self.M

        X: vecteur de donnees
        t: vecteur de cibles
        Mmin : plus petite valeur de M à tester
        Mmax : plus grande valeur de M à tester

        """
        t = np.expand_dims(t, axis=1)
        liste_m = []
        liste_erreurs = []
        taille_validation = int(np.round(X.shape[0]/5))
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
                error_val_total += self.erreur(t_val, t_pred_val)
            erreur_val_moy = error_val_total / 10
            liste_m.append(M)
            liste_erreurs.append(erreur_val_moy)
        index_best_m = liste_erreurs.index(min(liste_erreurs))
        self.M = liste_m[index_best_m]
        print("Meilleure valeur de l'hyperparamètre M trouvé lors de la 10-fold cross-validation : {}".format(self.M))

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
        if self.M <= 0:
            self.recherche_hyperparametre(X, t, Mmin=1, Mmax=10)

        phi_x = self.fonction_base_polynomiale(X, using_sklearn=using_sklearn)

        if not using_sklearn:
            self.w = np.linalg.solve((phi_x.T.dot(phi_x) + np.identity(phi_x.shape[1]) * self.lamb), (phi_x.T.dot(t)))
            print('w trouvé : {}'.format(self.w))
        else:
            reg = linear_model.Ridge(alpha=self.lamb, fit_intercept=False)
            reg.fit(phi_x, t)
            self.w = reg.coef_
            print('w trouvé : {}'.format(self.w))
        return phi_x

    def prediction(self, x, using_sklearn):
        """
        Retourne la prediction de la regression lineaire
        pour une entree, representee par un tableau 1D Numpy ``x``.

        Cette methode suppose que la methode ``entrainement()``
        a prealablement ete appelee. Elle doit utiliser le champs ``self.w``
        afin de calculer la prediction y(x,w) (equation 3.1 et 3.3).
        """
        phi_x = self.fonction_base_polynomiale(x, using_sklearn=using_sklearn)
        prediction = phi_x.dot(self.w)
        return prediction

    @staticmethod
    def erreur(t, prediction):
        """
        Retourne l'erreur de la difference au carre entre
        la cible ``t`` et la prediction ``prediction``.
        On ne considère pas la régression avec prédictions multiples ( donc prediction et p sont des vecteurs à une dimension )
        """

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
    #test_fonction_recherche_hyperparametres()

    def test_entrainement():
        reg = Regression(lamb=0, m=0)
        w = [0.3, 4.1]
        modele_gen = "tanh"
        nb_train = 1000
        nb_test = 100
        bruit = 0.2
        gestionnaire_donnees = gd.GestionDonnees(w, modele_gen, nb_train, nb_test, bruit)
        [x_train, t_train, x_test, t_test] = gestionnaire_donnees.generer_donnees()
        print("x", x_train.shape)
        phi_x = reg.entrainement(x_train, t_train, using_sklearn=True)
        t_pred_train = phi_x.dot(reg.w)
        print(t_pred_train.shape)
        phi_x_test = reg.fonction_base_polynomiale(x_test)
        t_pred_test = phi_x_test.dot(reg.w)
        plt.subplot(211)
        plt.title('Donnees d\'entrainement')
        plt.scatter(x_train, t_train)
        plt.scatter(x_train, t_pred_train)
        plt.subplot(212)
        plt.title('Donnees de test')
        plt.scatter(x_test, t_test)
        plt.scatter(x_test, t_pred_test)
        plt.show()
    #test_entrainement()

    def test_fonction_base_polynomiale_scikit():
        reg = Regression(lamb=5, m=5)
        x_int = 2
        x = np.array([1, 2, 3, 4])

        phi_x = reg.fonction_base_polynomiale(x, False)
        phi_x_true = reg.fonction_base_polynomiale(x, True)
        print("Fonction sans scikit learn \n", phi_x)

        print(type(reg.fonction_base_polynomiale(x, False)[0, 0]))
        print(reg.fonction_base_polynomiale(x, False).shape)

        print("Fonction avec scikit learn \n ", phi_x_true)

        print(type(reg.fonction_base_polynomiale(x, True)[0, 0]))
        print(reg.fonction_base_polynomiale(x, True).shape)
    #test_fonction_base_polynomiale_scikit()

