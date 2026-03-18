# Rapport technique — Puls Events RAG

## 1. Résumé exécutif

Ce projet consiste à concevoir un **POC (Proof of Concept)** de système **RAG (Retrieval-Augmented Generation)** pour la recommandation d'événements culturels.  
L'objectif métier est de démontrer qu'une plateforme telle que **Puls-Events** peut intégrer un assistant capable de répondre à des questions utilisateurs en s'appuyant sur une base d'évC�nements structurée, vectorisée et interrogeable sémantiquement.

Le système final combine :

- une ingestion configurable de données d'événements ;
- un pipeline de préparation et de transformation documentaire ;
- un index vectoriel **FAISS** ;
- une chaîne **LangChain** ;
- des modèles **Mistral** pour les embeddings et la génération ;
- une API **FastAPI** ;
- une stratégie d'évaluation automatisée ;
- une exécution locale, scriptable et conteneurisée.

Le projet a été construit pour répondre aux attentes de la mission OpenClassrooms **Concevez et déployez un système RAG**.

---

## 2. Contexte et besoin métier

L'entreprise fictive **Puls-Events** souhaite tester un chatbot capable de répondre  à des questions sur des évC�nements culturels à venir ou récents.

Exemples de besoins métier :

- recommander des événements pertinents en fonction d'une requête libre ;
- résumer rapidement l'offre culturelle locale ;
- fournir une réponse formulée naturellement, sans obliger l'utilisateur à parcourir une liste brute d'événements ;
- exposer une API exploitable par des équipes produit ou marketing.

Le POC doit donc démontrer :

- la **faisabilité technique** ;
- la **pertinence fonctionnelle** ;
- la **capacité dévaluation** ;
- la **facilité d'éntégration** via API.

---

### 3. Périmètre retenu

Le cahier des charges autorise un choix libre de zone géographique, à condition de conserver un périmètre cohérent et des événements récents ou à venir.

Le périmètre retenu pour le développement principal est :

- **Ville** : Montpellier
- **Mode temporel** : `rolling`
- **Fenêtre temporelle** : 365 jours

Ce choix a été motivé par un compromis entre :

- richesse suffisante du corpus ;
- temps de traitement raisonnable ;
- démonstration claire du système ;
- coût d'évaluation limité.

Lors des essais, un filtrage plus étroit ou au contraire trop large s'est révélé moins adapté au format POC. Le corpus final de travail contient environ **688 évC�nements**, ce qui reste maniable tout en fournissant une diversité utile.

---

## 4. Source de données et justification

### 4.1. Source utilisée

Le projet s'appuie sur le dataset public OpenDataSoft :

- **Nom** : `evenements-publics-openagenda`
- **Endpoint** : `/api/explore/v2.1/catalog/datasets/evenements-publics-openagenda/records`

### 4.2. Justification du choix

L'énoncé mentionne OpenAgenda, mais les ressources du projet fournissent explicitement le dataset OpenDataSoft exposant ces événements.  
Le choix d'OpenDataSoft a été retenu pour les raisons suivantes :

- accès public direct ;
- mise en öuvre reproductible ;
- absence de dépendance à une clé OpenAgenda dédéeé ;
- meilleure conformité pratique.

gCe choix reste aligné avec le besoin métier : exploiter des évC�nements publics issus de l'écosystème OpenAgenda pour construire un assistant de recommandation.

---

## 5. Schéma UML / architecture

## Schéma UML simplifié

``mermaid
flowchart TD
    A[OpenDataSoft events dataset] --> B[build_dataset.py]
    B --> C[data/raw/events.json]
    C --> D[build_index.py]
    D --> E[Chunking]
    E --> F[Mistral Embeddings]
    F --> G[FAISS Index]
    G --> H[Retriever]
    H --> I[LangChain RAG Service]
    I --> J[FastAPI /ask]
    D --> K[FastAPI /rebuild]
    I --> L[evaluate_rag.py]
    I --> M[evaluate_ragas.py]
```

### Description des composants

- **OpenDataSoft client** : récupàre les évC�nements via l'API REST publique
- **Preprocessing** : nettoie é*reformate les données en documents textuels
- **Chunking** : segmente les documents pour la vectorisation
- **Embeddings Mistral** : transforme les chunks en vecteurs sémantiques
- **FAISS** : stocke les vecteurs et permet la recherche par similarit�
- **Retriever** : récupère les chunks les plus proches d'une question
- **Service RAG** : construit le contexte et interroge le modèle de génération
- **FastAPI** : expose les endpoints métier
- **Scripts dévaluation** : mesurent la qualité du système
- **Docker / CI** : assurent la reproductibilité et l'automatisation

---

## 6. Environnement technique

### Langage et runtime

- Python 3.12

### Bibliothèques principales

- FastAPI
- Uvicorn
- LangChain
- langchain-community
- langchain-mistralai
- langchain-text-splitters
- faiss-cpu
- pydantic-settings
- structlog
- httpx

### Outils de qualité

- uv
- pytest
- Ruff
- Bandit
- pre-commit
- GitHub Actions

### Outils dévaluation

- ragas
- datasets

### Déploiement

- Docker

---

### 7. Organisation du dépôt

Le dépôt a été structuré de manière à séparer clairement :

- la logique métier ;
- les scripts d'exécution ;
- les tests ;
- les données ;
- la documentation ;
- les workflows CI.

Arborescence simplifiée :

 ```text
src/puls_events_rag/
├─ api/
├─ evaluation/
“─ ingestion/
“─ rag/
“─ config.py
├─ logger.py

scripts/
├─ build_dataset.py
├─ build_index.py
├— evaluate_rag.py
├─ evaluate_ragas.py
├— api_test.py
├─ run_local.py

tests/
├─ unit/
“─ integration/
```

Cette structure permet à un évaluateur ou collègue de retrouver rapidement les différentes couches du système.

---

## 8. Gestion de configuration

La configuration est centralisée dans `config.py` via **Pydantic Settings** et un fichier `.env`.

Les principaux paramètres configurables sont :

- champ géographique (`city`, `region`, `department`)
- valeur géographique
- langue
- fuseau horaire
- mode temporel (`past` ou `future` ou `rolling`)
- fênêtre en jours
- batch size d'ingestion
- limite max de corpus
- modèle d'embedding
- modèle de génération
- paramètres de chunking
- nom de l'index FAISS
- token de protection de `/rebuild`

Ce choix rend le système :

- flexible ;
- relançable ;
- facilement démontrable avec plusieurs périmètres.

---

## 9. Ingestion des données

### 9.1. Objectif

Récupérer les événements de manière fiable, filtrée et relançable.

### 9.2. Mécanisme

Le client OpenDataSoft construit une clause `where` dynamique combinant :

- le filtre géographique ;
- la contrainte temporelle.

Exemple de logique :

- `location_city = 'Montpellier'`
- `firstdate_begin >= date'2025-03-11'`
- `firstdate_begin <= date'2026-03-11'`

L'ingestion se fait par pagination (`limit`, `offset`) avec accumulation de tous les résultats jusqu'à :

- épisement des pages
- ou atteinte de `ingestion_max_records`

### 9.3. Contrôles

Des tests unitaires vérifient :

- la construction de la clause `where`
- le nettoyage HTML
- le mapping d'un événement vers un document RAG

---

### 10. Prétraitement des données

Chaque évC�nement récupáré st transformé en document textuel structuré contenant notamment :

- titre
- description courte
- description longue nettoyée
- conditions
- ville
- lieu
- adresse
- date
- mots-clés

Le nettoyage HTML est volontairement simple mais robuste pour le POC :

- suppression des balises
- normalisation des espaces
- décodage des entités HTML

Le résultat est sauvegardé dans `data/raw/events.json`.

---

### 11. Chunking

### 11.1. Pourquoi chunker ?

Les descriptions d'événements peuvent être longues et hétérogènes.  
Le chunking permet :

- une vectorisation plus stable ;
- une meilleure granularité en retrieval ;
- une limitation de la taille des contextes transmis au LLM.

### 11.2. Implémentation retenue

Le projet utilise `RecursiveCharacterTextSplitter` avec des séparateurs hiérarchiques :

- `\n\n`
- `\n`
- `. `
- ` `
- `""`

Paramètres par défaut :

- `chunk_size = 800`
- `chunk_overlap = 120`

### 11.3. Alternatives possibles

D'autres strat�gies auraient pu  être envisagées :

- chunking sémantique ;
- découpage par sections de métadonnées ;
- phrase splitting ;
- sliding window plus dense.

Pour un POC, le splitter récursif offre un trés bon compromis entre simplicité, lisibilité et efficacité.

---

## 12. Embeddings

### 12.1. Choix du modèle

Le projet utilise **MistralAIEmbeddings** avec le modèle :

- `mistral-embed`

### 12.2. Justification

Ce choix a été retenu pour :

- la cohérence avec les consignes du projet ;
- l'homégénéité de la stack Mistral ;
- l'éntégration native avec LangChain ;
- la simplicité d'éxploitation.

### 12.3. Limites
L'usage d'un service externe implique :

- dépendance réseau ;
- dépendance à une clé API ;
- coût éventuel ;
- variabilité potentielle des performances selon l'API distante.

This limite est explicitement assumée dans le cadre du POC.

---

## 13. Base vectorielle FAISS

### 13.1. Rôle

FAISS stocke les vecteurs d'embeddings et permet la recherche sémantique rapide des chunks les plus proches.

### 13.2. Choix de FAISS

FAISS a été choisi car :

- il est largement utilisé ;
- il s'intègre facilement avec LangChain ;
- `faiss-cpu` est portable ;
- il est très adapté à un POC local.

### 13.3. Persistance

L'index est sauvegardé localement dans :

- `index.faiss`
- `index.pkl`
- `manifest.json`

Le `manifest.json` contient notamment :

- le nom de l'index
- le modèle d'embedding
- le nombre de documents
- le nombre de chunks
- les paramètres de chunking

---

## 14. Retrieval

### 14.1. Principe

Lorsqu'une question utilisateur est reçue, le système :

1. charge l'index FAISS
2. transforme la question en embedding
3. récupàre les `k` documents les plus proches par similarité

### 14.2. Paramétrage

- `retrieval_k` = 5 par défaut

Ce Choix permet d'obtenir :

- assez de contexte pour répondre ;
- sans surcharger inutilement le prompt.

### 14.3. Limites

Le système ne comporte pas de **reranker** dédi。

Le point a été laissé en amélioration potentielle, car il nétait pas requis for the mission et aurait ajouté de la complexité à un POC dont l'objectif principal était la démonstration de faisabilité.

---

## 15. Génération de réponse

### 15.1. Modèle

Le projet utilise **ChatMistralAI** avec :

- `mistral-small-latest`

### 15.2. Prompting

Le prompt système impose plusieurs contraintes :

- répondre uniquement à partir du contexte fourni ;
- ne pas inventer d'information ;
- restituer les informations utiles ;
- répondre en français ;
- signaler explicitement l'absence d'information si nécessaire.

Le contexte est construit à partir des chunks récupárés, enrichis de métadonnés :

- titre
- ville
- date
- lieu
- adresse
- conditions

### 15.3. Sources

La réponse renvoie également une liste structurée de sources :

- `uid`
- `title`
- `city`
- `date`
- `score` (actuellement non utilisé)

Les doublons sont filtrés au niveau du service.

---

## 16. API REST

LAPI a ét� implémentée avec **FastAPI**.

### Endpoints disponibles
#### `GET /health`

Permet de vérifier que le service est disponible.

#### `GET /metadata`

Expose la configuration métier du corpus courant.

#### `POST /ask`

Entrée :

- `question`
- `top_k`

Sortie :

- `answer`
- `sources`

#### `POST /rebuild`

Reconstruit l'index vectoriel.  
Cet endpoint est protégé par un token transmis dans le corps de la requête.

### Justification

FastAPI a été retenu car il fournit :

- validation automatique des schémas ;
- documentation Swagger ;
- simplicité de mise en Œuvre ;
- bonne lisibilit� pour un POC.

---

## 17. Qualité logicielle et tests

### 17.1. Tests unitaires

Des tests ont été ajoutés pour valider :

- la clause de filtrage OpenDataSoft
- le nettoyage HTML
- la transformation évC�nement → document
- la création de documents LangChain
- le chunking
- certaines fonctions d'évaluation
- les réponses API de base

### 17.2. Tests d'intégration

Des tests d'éntégration vérifient notamment :

- `/health`
- `/metadata`
- la validation de `/ask`

### 17.3. Script de test API

Un script `scripts/api_test.py` permet de tester rapidement :

- `/health`
- `/metadata`
- `/ask`

Ce script est particulièrement utile pour la démonstration.

---

## 18. Strat�gie d'évaluation

La qualité du système a été évaluée selon deux approches complémentaires.

### 18.1. Évaluation heuristique

Un jeu de test annoté simple (`reference_qa.json`) a été constitué, avec pour chaque question :

- une réponse de référence ;
- des mots-clés attendus ;
- une ville attendue.

Le script `evaluate_rag.py` mesure :

- la couverture de mots-clés ;
- la cohérence géographique des sources ;
- un label global :
  - `correct`
  - `partially_correct`
  - `incorrect`

#### Intérêt

- relançable rapidement ;
- simple à interpréter ;
- adapté à la non-régression.

#### Limite

- assez indulgent ;
- sensible au choix des mots-clés ;
- peu fin sur la qualité réelle du langage produit.

### 18.2. Évaluation Ragas

Le script `evaluate_ragas.py` évalue le système avec :

- `faithfulness`
- `answer_relevancy`
- `context_precision`

L'évaluation Ragas est configurée avec :

- **ChatMistralAI** comme LLM d'évaluation ;
- **MistralAIEmbeddings** pour les embeddings.

Se oprojet reste donc **full Mistral**, sans dépendance à OpenAI.

### 18.3. Interprétation des résultats

Les scores obtenus sont bons sur le jeu de test construit, ce qui montre :

- que le pipeline fonctionne ;
- que le contexte récupéré est généralement cohérent ;
- que les réponses restent globalement alignées avec les données.

Cependant, ces résultats ne doivent pas être surinterprétés.  
Le jeu de test est encore réduit, et l'évaluation reste limitée par :

- la taille du benchmark ;
- l'absence d'annotation humaine multi-niveaux ;
- l'absence de cas adversariaux plus durs.

---

## 19. CI / automatisation

Deux workflows GitHub Actions ont asté ajoutés :

### `ci.yml`

Exécute :

- `uv sync --all-groups`
- `ruff check`
- `ruff format --check`
- `bandit`
- `pytest`

### `eval.yml`

Permet de relancer à la demande :

- la reconstruction du dataset ;
- la reconstruction de l'index ;
- l'évaluation heuristique ;
- l'évaluation Ragas

Cette automatisation répond à l'exigence de relançabilité du projet.

---

### 20. Script de lancement local

Le script `run_local.py` permet d'enchaîner automatiquement :

1. `build_dataset.py`
2. `build_index.py`
3. `evaluate_rag.py`
4. lancement de l'API

Ce script facilite :

- la reproduction complète ;
- la démonstration ;
- la vérification rapide du pipeline de bout en bout.

---

## 21. Déploiement local avec Docker

Un `Dockerfile` est fourni afin de construire une image locale de l'API.

### Intérêt

- reproductibilité ;
- démonstration locale ;
- préparation à un déploiement élargi.

### Limite

Même conteneurisé, le système dépend toujours de l'API Mistral pour :

- les embeddings ;
- la génération ;
- certaines évaluations.

Il n'est donc pas entièrement autonome hors ligne.

---

## 22. Sécurité et robustesse

### Mesures prises

- clé API stockée dans `.env`
- `.env` ignoré par Git
- validation Pydantic des requêtes
- endpoint `/rebuild` protégé par token
- analyse Bandit
- tests automatisés
- gestion structurée heure logs

### Limites
Le projet reste un POC local.  
Si une exposition publique devait être envisagée, il faudrait renforcer :

- authentification / autorisation ;
- gestion plus fine des erreurs ;
- rotation des secrets ;
- supervision ;
- politiques de rate limiting.

---

## 23. Forces du projet

- pipeline complet de bout en bout ;
- architecture claire ;
- stack cohérente avec le cahier des charges ;
- séparation logique métier / API ;
- scripts relançables ;
- évaluation présente à deux niveaux ;
- documentation reproductible ;
- conteneurisation disponible.

---

## 24. Limites du projet

- jeu de test encore modeste ;
- pas de reranking ;
- pas de mémoire conversationnelle ;
- dépendance réseau à Mistral ;
- èvaluation heuristique encore simplifiée ;
- pas de front dédié ;
- pas de déploiement cloud finalisé.

---

## 25. Perspectives d'amélioration
- enrichir le benchmark annoté ;
- ajouter une évaluation humaine détailée;
- introduire un reranker ;
- améliorer le scoring et la citation des sources ;
- filtrer plus finement par catégories d'événements ;
- intégrer une interface utilisateur ;
- renforcer la supervision et l'observabilité ;
- préparer un déploiement managé.

---

### 26. Conclusion

Le projet aboutit è un **POC RAG cohérent, fonctionnel et démontrable**, répondant aux principales attentes du cahier des charges :

- répération de données d'événements ;
- préparation documentaire ;
- chunking ;
- embeddings ;
- indexation vectorielle FAISS ;
- retrieval ;
- génération ;
- API REST ;
- endpoint de reconstruction ;
- tests ;
- évaluation ;
- CI ;
- lancement local ;
- conteneurisation Docker.

Le système est suffisamment abouti pour être présenté en soutenance comme une preuve de faisabilité crédible, tout en laissant apparaét honnêtement les limites normales d'un POC et les pistes d'industrialisation futures.