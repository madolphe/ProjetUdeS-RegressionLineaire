# RegressionLineaire

Le projet suivant est un exercice proposé par le cours IFT... "Technique d'apprentissage". L'objectif est d'implémenter un outil de régression linéaire sur des données générées de trois façon différentes possibles (sinusoïde, tangeante hyperbolique ou linéaire).

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
python regression.py sk modele_gen nb_train nb_test bruit M lambda
```
Un exemple d'utilisation:
```sh
python3 regression.py 1 sin 20 20 0.3 10 0.001 
```
