import sys
import os

# Obtenez le chemin du répertoire contenant vos fichiers
directory_path = 'Lanouraf/projet-de-deep-learning-'

# Ajoutez le répertoire au chemin Python
sys.path.append(directory_path)

# Vérifiez à nouveau si le chemin Python contient ce répertoire
if directory_path in sys.path:
    print("Le répertoire est maintenant dans le chemin Python.")
else:
    print("Le répertoire n'est toujours pas dans le chemin Python.")
