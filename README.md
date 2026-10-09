# 🌦️ Robustesse de la détection d'objets face à la météo

Une caméra de contrôle routier travaille dehors, par tous les temps. Ce projet mesure et compare **à quel point deux détecteurs d'objets entraînés par beau temps perdent leurs moyens** (YOLOv8, un CNN, et RT-DETR, un transformer) quand la météo se dégrade : brouillard, nuit, pluie, neige et éblouissement, chacun à 5 niveaux d'intensité.

> Projet réalisé dans le cadre de ma formation à l'ESMT Dakar, refait et documenté en 2026.

![YOLOv8s contre RT-DETR-l face à la météo](assets/comparaison_modeles.png)

## 🎯 Problème métier
Les systèmes de vidéo-verbalisation doivent repérer le **téléphone au volant** et le **défaut de ceinture**, en plus de la plaque et du véhicule. Un modèle excellent au soleil ne sert à rien s'il devient aveugle sous la pluie. La question posée : **quelle météo, et à partir de quel niveau, rend la détection inutilisable ?**

## 📊 Données
- **500 images** de caméras de contrôle routier, vues de face, annotées en boîtes (format YOLO).
- **5 classes** : `license_plate` (plaque), `phone` (téléphone en main), `vehicle` (véhicule), `seatbelt` (ceinture), `windshield` (pare-brise).

![Exemples de chaque classe](assets/classes.jpg)

- Chaque image a été dégradée artificiellement par **5 conditions météo × 5 niveaux**, soit **26 versions** du jeu de données (13 000 images avec l'original).
- Le jeu de données n'est pas publié dans ce dépôt (1,2 Go).

## 🧪 Protocole
1. Découpage des 500 images originales : **350 entraînement / 75 validation / 75 test** (graine 42).
2. Entraînement de deux modèles pré-entraînés sur COCO, **uniquement sur les images originales** : 50 époques, 640 px, GPU T4 (Google Colab).
   - **YOLOv8s** (CNN, 11,1 M paramètres) : 8 minutes d'entraînement.
   - **RT-DETR-l** (transformer, 32,8 M paramètres) : 30 minutes d'entraînement.
3. Évaluation de chaque modèle sur **les mêmes 75 images de test**, dans chacune des 26 conditions.
4. Métriques : **mAP@50** et **mAP@50-95**.

## 📈 Résultats

### Par temps normal (validation, 75 images, 232 objets), mAP@50

| Classe | YOLOv8s | RT-DETR-l |
|--------|---------|-----------|
| **Toutes** | **0,787** | **0,781** |
| Pare-brise | 0,995 | 0,995 |
| Plaque | 0,988 | 0,994 |
| Véhicule | 0,747 | 0,850 |
| Téléphone | 0,691 | 0,500 |
| Ceinture | 0,516 | 0,567 |

Les deux modèles se valent par beau temps. Les petits objets (téléphone, ceinture) sont les plus difficiles pour les deux.

### Face à la météo (test, mAP@50 au niveau 5, le plus extrême)

| Météo | YOLOv8s | RT-DETR-l |
|-------|---------|-----------|
| Aucune (original) | 0,777 | 0,807 |
| ❄️ Neige | 0,129 (-83 %) | **0,427 (-47 %)** |
| 🌧️ Pluie | 0,317 (-59 %) | **0,371 (-54 %)** |
| 🌫️ Brouillard | 0,581 (-25 %) | **0,670 (-17 %)** |
| 🌙 Nuit | 0,623 (-20 %) | 0,622 (-23 %) |
| ☀️ Éblouissement | 0,771 (-1 %) | 0,810 (0 %) |

Tous les niveaux, avec le mAP@50-95 : [`results/resultats_yolov8s.csv`](results/resultats_yolov8s.csv) et [`results/resultats_rtdetr-l.csv`](results/resultats_rtdetr-l.csv).

### Le prix de la robustesse (GPU T4)

| | YOLOv8s | RT-DETR-l |
|---|---|---|
| Paramètres | 11,1 M | 32,8 M |
| Entraînement (50 époques) | 8 min | 30 min |
| Inférence par image | ≈ 11 ms | ≈ 43 ms |

## 💡 Ce qu'il faut retenir
- **La neige est la pire condition pour les deux modèles**, mais c'est là que l'écart est le plus grand : au niveau 5, RT-DETR garde **3 fois plus** de mAP@50 que YOLOv8 (0,427 contre 0,129).
- **RT-DETR résiste mieux** à la neige, à la pluie et au brouillard. Une hypothèse : son attention globale (transformer) s'appuie sur le contexte de toute l'image quand les détails locaux disparaissent.
- **La nuit les touche autant l'un que l'autre** (0,62 au niveau 5).
- **L'éblouissement n'a aucun effet mesurable** : la perturbation générée est probablement trop légère. C'est une limite du jeu de données, pas une preuve de robustesse.
- **Ce gain se paie** : RT-DETR est 3 fois plus gros et environ 4 fois plus lent. Pour une caméra embarquée, YOLOv8 reste plus réaliste ; pour un traitement sur serveur, RT-DETR est plus fiable par mauvais temps.

## ⚠️ Limites
- Jeu de test de 75 images : de petits écarts (quelques centièmes) ne sont pas significatifs. Par exemple, RT-DETR fait 0,819 au brouillard de niveau 1, un peu plus qu'au départ (0,807), ce qui relève du bruit.
- Seulement **8 téléphones** en validation : le score de cette classe est fragile.
- Les dégradations sont **synthétiques** : de vraies images de pluie ou de neige pourraient donner d'autres résultats.
- Une seule graine d'entraînement par modèle, pas encore de barres d'erreur.

## 🔜 Prochaines étapes
- Ajouter **Faster R-CNN** pour compléter la comparaison.
- Tester un entraînement **avec augmentation météo** pour voir si YOLOv8 rattrape RT-DETR.
- Analyser la chute **par classe** (téléphone et ceinture en particulier).

## 🛠️ Stack
Python · Ultralytics (YOLOv8, RT-DETR) · PyTorch · pandas · Matplotlib · Google Colab (GPU T4)

## ▶️ Reproduire
1. Mettre le dossier du jeu de données zippé (`results.zip`) à la racine de Google Drive.
2. Ouvrir [`notebooks/robustesse_meteo.ipynb`](notebooks/robustesse_meteo.ipynb) dans Colab, choisir un GPU T4, puis exécuter les cellules dans l'ordre (sections 1 à 5 pour YOLOv8, 6 et 7 pour RT-DETR et la comparaison).

---
👩🏾‍💻 **Gnilane Faye**, élève-ingénieure IA & Data, ESMT Dakar · [LinkedIn](https://www.linkedin.com/in/gnilanefaye) · [GitHub](https://github.com/gnilanefaye)
