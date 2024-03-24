# ProjetDL

### Sources
* Article :
  - Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift: [Article](https://arxiv.org/abs/1502.03167)
  - Layer Normalization: [Article](https://arxiv.org/abs/1607.06450)
  - Code source associé à la Batch Normalisation : [GitHub](https://github.com/David-Estevez/BatchNorm/tree/master/BatchNorm)
* Overleaf :
  - Lien vers le document sur Overleaf : [Overleaf](https://fr.overleaf.com/read/szxdnkrtxrhr#4172b5)

### Description
Ce projet vise à explorer et à comparer les performances de la normalisation par lot (Batch Normalization) et de la normalisation par couche (Layer Normalization) dans différents contextes d'application en Deep Learning.

### Application Streamlit
Nous avons développé une application Streamlit pour présenter les performances de la Batch Normalization et de la Layer Normalization dans différents cas d'applications. Pour lancer cette application, veuillez exécuter la commande suivante :
```bash
streamlit run app.py
```
L'application présente une analyse comparative des deux techniques de normalisation dans divers scénarios d'utilisation, en mettant en évidence leurs avantages et leurs limitations.

Les bibliothèques requises pour exécuter l'application sont répertoriées dans le fichier requirements.txt. Pour les installer, veuillez exécuter la commande suivante :
```bash
pip install -r requirements.txt
```