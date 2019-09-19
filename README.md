# RegressionLineaire

Le projet suivant est un exercice proposé par le cours IFT712 "Technique d'apprentissage". L'objectif est d'implémenter un outil de régression linéaire sur des données générées de trois façon différentes possibles (sinusoïde, tangeante hyperbolique ou linéaire).

# Installation

Requis: Python 3.5+ | Linux, Mac OS X, Windows

```sh
pip install pipenv
```
Puis dans le dossier du projet:  

```sh
pipenv install --python 3.5
```
Le pipfile permettra l'installation de toutes les dépendances nécessaires à l'utilisation du projet. 
Puis pour executer des commandes dans cet environnement virtuel: 

```sh
pipenv shell
```

# Getting Started

Pour lancer le script principal, le passage des arguments se fait de la manière suivante: 
```sh
python regression.py sk type_de_modele nb_train nb_test bruit M lambda
```
Un exemple d'utilisation:
```sh
python3 regression.py 1 sin 20 20 0.3 10 0.001 
```

</br>

En plus de l'exercice de développement d'un outil de régression, un script permettant une autre visualisation est proposée dans ce projet. En effet, pour effectuer des regressions non linéaires, une fonction de base polynomiale est utilisée pour augmenter la dimensionnalité de la donnée initiale. Dans ce cadre, la regression linéaire est effectué dans un espace de plus grande dimension (elle correspond à faire passer un hyperplan dans ce nouvel espace). Le script visualisation3D permet de "voir" le calcul des paramètres de l'hyperplan dans un espace de dimension 3.

</br>

Un exemple d'utilisation:
```sh
python3 visualisation3D.py sin 200 0.3 0.001
```

# Exemples script principal
Le lancement du script principal ouvre séquentiellement 2 fenêtres. Tout d'abord, un graphique présente les résultats de la recherche d'hyperparamètre (si celle-ci a été passé en argument):
![alt text](https://github.com/EmmaJouffroy/RegressionLineaire/blob/master/extra/recherche-hyperparams.png)
Ensuite, un graphique montre le modèle trouvé :
![alt text](https://github.com/EmmaJouffroy/RegressionLineaire/blob/master/extra/exemple-modele.png)

# Exemples script visualisation 3D
Le script de visualisation 3D permet de voir la régression linéaire effectuée dans un espace de dimension 3. </br>
Voici un exemple avec des données générées linéairement (envoie dans un espace de dimension supérieur inutile mais l'idée reste la même): 
![alt text](https://github.com/EmmaJouffroy/RegressionLineaire/blob/master/extra/gif-ex-lin\)visu3D.gif)




