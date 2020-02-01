# Outils utilisés pour l'étude STA211

## Pré-requis
`Python 3.5+` et `pip`

## Installation des dépendences
`pip3 install -r requirements.txt`

## Preprocessing
`python3 preprocessing.py`

## Entraînement
`python3 train.py`

## Tester un modèle simple
Liste des modèles dans le fichier `sta211/selection.py`

Ces fonctions renvoient toutes un tableau d'estimators à mettre dans un [Pipeline](https://scikit-learn.org/stable/modules/compose.html#combining-estimators) et les paramètres pour la [recherche en grille](https://scikit-learn.org/stable/modules/grid_search.html#exhaustive-grid-search)
```
get_svm()
get_mlp()
get_gradient_boosting()
get_random_forest()
get_extra_trees()
get_adaboost()
get_naive_bayes()
```

Modifier les variables pipes & search_grid dans `train.py`
```
# Ex pour SVM
pipes, search_grid = get_svm()
pipe = Pipeline(pipes)
...
grid = GridSearchCV(pipe, search_grid, cv=cv, n_jobs=n_jobs, return_train_score=True, refit=True, scoring="accuracy")

```

## Tester une agrégation de modèles
Utiliser la fonction `get_voting_classifier` dans `sta211/selection.py` qui renvoie le Pipeline déjà rempli
```
# Ex pour agrégation manuelle
pipe, search_grid = get_voting_classifier()
...
grid = GridSearchCV(pipe, search_grid, cv=cv, n_jobs=n_jobs, return_train_score=True, refit=True, scoring="accuracy")

```
