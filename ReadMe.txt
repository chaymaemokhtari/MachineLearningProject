***************** L'application Web *****************
-l'utilisateur doit entrer en min les donnes de 5 etudiants pour des bons resultats.

-L'application web comporte 5 pages 
	----> HomePage
	----> prediction 1 : affiche les resultats de predition de la note de la premiere periode en utilisant
			les differents algos de classification et de regression
	----> prediction 2 : affiche les resultats de predition de la note de la troisieme periode en utilisant
			les differents algos de classification et de regression
	----> Student Form  : formulaire pour la saisie de differents features et targets
	----> DataSet   : Tableau representatif des données saisies ( avec boutton clear pour renitialiser le dataset)

-pour lancer l'applicaion
	executer la commande   "flask run" dans le terminal apres avoir navigue vers le dossier du projet

dans le cas du non fonctionement de la commande flask run 
set FLASK_APP=app.py
$env:FLASK_APP = "app.py"
python -m  flask run


******************** Notebooks *******************************
Il y a deux prédictions qui sont faites:
- dans la première prédiction, G1 est le target, la régression linéaire est appliquée dans le notebook "LinearRegression1", alors que les autres algorithmes de classifications sont appliqués dans le fichier "Classification1".
- dans la 2ème prédiction, G3 est le target, G1 fait parti des features, la régression linéaire est appliquée dans le notebook "LinearRegression2", alors que les autres algorithmes de classifications sont appliqués dans le fichier "Classification2". 
On a gardé G3 comme target et on s'est limité sur G1 comme feature dans le fichier "LinearRegression3" dans lequel la régression linéaire est appliquée.

* student-mat : dataset sous forme de fichier csv 
* Models1 : contient les modèles de la 1ère prédiction 
* Models2 : contient les modèles de la 2ème prédiction(celle avec plusieurs features)
* Models3: contient le modèle de la régression linéaire de la 2ème prédiction(celle avec seulement G1 comme feature) 
* df_class1: dataset pour la 1ère classification
* df_class2: dataset pour la 2ème classification
* df_linreg: dataset pour la régression linéraire 
* student.txt : fichier qui donne plus d'informaions sur notre dataset 
