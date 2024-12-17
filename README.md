L'implémentation de ce projet repose sur l'utilisation d'un ensemble d'outils et de bibliothèques permettant d'analyser des images et de détecter les émotions faciales en temps réel. Grâce à des modèles de réseaux de neurones convolutifs (CNN) et à l'utilisation de bibliothèques performantes comme TensorFlow, Keras, et OpenCV, le système est capable de capturer, traiter et interpréter des expressions faciales. L'interface utilisateur a été conçue avec Tkinter pour offrir une interaction fluide avec le modèle de détection, facilitant ainsi la gestion des résultats, des images capturées, et des différentes analyses

L'interface est structurée autour d'une fenêtre principale divisée en plusieurs sections :
* *Zone de détection* : Un cadre où l'image ou le flux vidéo capturé par la webcam est affiché.
* *Historique des émotions* : Un panneau affichant les émotions détectées au fur et à mesure, sous forme d'une liste numérotée.
* *Historique des images capturées* : Un autre panneau où chaque image capturée est affichée avec une taille uniforme.
* *Boutons d'action* : Des boutons placés en haut de l'interface pour contrôler les fonctions de détection.

Boutons d'Action :
* *Démarrer Détection* : Ce bouton lance la capture vidéo et active la détection des émotions en temps réel.
* *Arrêter Détection* : Il arrête la détection et la capture vidéo.
* *Détecter* : Ce bouton permet de déclencher manuellement la détection des émotions pour une image sélectionnée ou une image capturée en temps réel.
* *Changer l'Expression* : Ce bouton permet de sélectionner une image de l'historique et de changer ses traits faciaux en fonction d'une émotion choisie.
* *À propos* : Ce bouton ouvre une petite fenêtre pour afficher des détails sur l'image sélectionnée dans l'historique, y compris la date, l'heure, et l'émotion détectée.
