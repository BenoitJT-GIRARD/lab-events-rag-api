# Rapport technique - Puls Events RAG

## 1. Objectifs du projet

### Contexte

Puls-Events souhaite tester un assistant intelligent capable de repondre a des questions en langage naturel sur des evenements culturels. Le POC doit montrer qu'une plateforme de recommandation peut s'appuyer sur un systeme RAG pour interroger un corpus d'evenements et retourner une reponse formulee naturellement.

### Problematique

Une liste brute d'evenements ou un moteur de recherche classique ne repond pas toujours bien a des questions libres du type :

- Quels evenements musicaux ont lieu a Montpellier ?
- Y a-t-il des evenements gratuits ?
- Quels evenements sont adaptes aux enfants ?

Un systeme RAG repond a ce besoin en combinant :

- une recherche semantique dans un corpus vectorise ;
- une generation de reponse naturelle fondee sur les documents retrouves ;
- une exposition simple via API pour les equipes produit et marketing.

### Objectif du POC

Le POC vise a demontrer :

- la faisabilite technique d'un systeme RAG complet ;
- la pertinence metier des reponses generees ;
- la possibilite de reconstruire l'index vectoriel a partir des donnees ;
- la capacite a evaluer automatiquement le systeme ;
- la possibilite d'exposer le systeme via une API REST reutilisable.

### Perimetre retenu

Le corpus est construit a partir du dataset public OpenDataSoft `evenements-publics-openagenda`.

Perimetre principal utilise pendant le developpement :

- localisation : `city = Montpellier`
- langue : `fr`
- fuseau : `Europe/Paris`
- mode temporel : `rolling`
- fenetre temporelle : `365` jours

Ce choix permet de conserver un corpus de taille raisonnable pour un POC local tout en gardant une diversite suffisante d'evenements.

## 2. Architecture du systeme

### Schema global

```mermaid
flowchart TD

    subgraph Data
        A[OpenDataSoft API]
        B[build_dataset.py]
        C[data/raw/events.json]
        A --> B --> C
    end

    subgraph Indexing
        D[build_index.py]
        E[LangChain Documents]
        F[Chunking]
        G[Mistral Embeddings]
        H[FAISS Index]
        C --> D --> E --> F --> G --> H
    end

    subgraph RAG
        I[Retriever]
        J[Prompt Template]
        K[ChatMistralAI]
        H --> I --> J --> K
    end

    subgraph API
        L[ask endpoint]
        M[rebuild endpoint]
        K --> L
        D --> M
    end

    subgraph Evaluation
        N[evaluate_rag.py]
        O[evaluate_ragas.py]
        L --> N
        L --> O
    end
```

### Role des composants

- OpenDataSoft / OpenAgenda : source de donnees d'evenements publics.
- Ingestion : recupere les evenements selon les filtres geographiques et temporels.
- Pretraitement : nettoie les champs HTML et reconstruit un texte metier exploitable.
- LangChain : orchestre documents, chunking, embeddings, vector store et modele de chat.
- MistralAIEmbeddings : transforme chaque chunk en vecteur semantique.
- FAISS : stocke les vecteurs et permet la recherche de similarite.
- Retriever : retrouve les `top_k` chunks les plus proches de la requete.
- ChatMistralAI : genere la reponse finale a partir du contexte retrouve.
- FastAPI : expose les endpoints metier et la documentation Swagger.
- Scripts d'evaluation : mesurent la qualite du systeme.

### Technologies utilisees

- Python 3.12
- FastAPI + Uvicorn
- LangChain
- langchain-mistralai
- langchain-community
- langchain-text-splitters
- FAISS CPU
- httpx
- pydantic-settings
- structlog
- pytest
- ragas
- Docker
- GitHub Actions

## 3. Preparation et vectorisation des donnees

### Source de donnees

Le projet utilise l'endpoint public OpenDataSoft expose pour le dataset OpenAgenda :

- dataset : `evenements-publics-openagenda`
- endpoint : `/api/explore/v2.1/catalog/datasets/evenements-publics-openagenda/records`

Parametres de filtrage utilises :

- `location_field` : `city`, `region` ou `department`
- `location_value` : valeur cible, par exemple `Montpellier`
- `date_window_mode` : `past`, `future` ou `rolling`
- `date_window_days` : taille de la fenetre temporelle
- `lang` et `timezone`

Le mode `rolling` est implemente comme une borne inferieure glissante sur `date_window_days`, sans borne superieure fixe.

### Nettoyage et structuration

Chaque evenement est transforme en document textuel contenant :

- titre
- description courte
- description longue nettoyee
- conditions
- ville
- lieu
- adresse
- date de debut
- mots-cles

Methodes appliquees :

- suppression des balises HTML
- normalisation des espaces
- decodage des entites HTML
- remplacement des valeurs manquantes par des chaines vides si necessaire

Ce choix permet d'obtenir un texte plus homogene pour l'embedding et le retrieval.

### Chunking

Le projet utilise `RecursiveCharacterTextSplitter` avec :

- `chunk_size = 800`
- `chunk_overlap = 120`
- separateurs : `\n\n`, `\n`, `. `, ` `, `""`

Justification :

- les descriptions d'evenements peuvent etre longues et heterogenes ;
- un chunking fin ameliore la granularite du retrieval ;
- l'overlap limite la perte d'information a la frontiere entre deux chunks ;
- le splitter recursif est simple, reproductible et bien adapte a un POC.

Autres strategies envisageables :

- chunking semantique ;
- chunking guide par la structure metier ;
- chunking par tokens ;
- parent-child retrieval.

Le choix retenu privilegie la simplicite de mise en oeuvre et la lisibilite du pipeline.

### Embedding

Le projet utilise `MistralAIEmbeddings` avec le modele :

- `mistral-embed`

Points importants :

- `MistralAIEmbeddings` est un wrapper LangChain pour l'API Mistral ;
- les chunks sont convertis en vecteurs denses de nombres reels ;
- les vecteurs sont ensuite indexes dans FAISS ;
- la requete utilisateur est encodee avec le meme modele pour rester dans le meme espace vectoriel.

Sur la dimension exacte des vecteurs, la source de verite est la documentation Mistral. Le code du repo ne fixe pas explicitement cette dimension : elle est geree par le service d'embedding du fournisseur.

La logique de batch est egalement delegatee au wrapper et a la bibliotheque sous-jacente. Pour ce POC, aucun tuning de batch n'a ete ajoute au niveau applicatif.

## 4. Choix du modele NLP

### Modeles selectionnes

Le systeme utilise deux composants Mistral distincts :

- embeddings : `mistral-embed`
- generation : `mistral-small-latest`

### Pourquoi ces modeles

Ces choix sont defensables pour un POC car ils offrent :

- une stack coherente chez un meme fournisseur ;
- une integration directe avec LangChain ;
- un cout et une complexite d'integration raisonnables ;
- une bonne adequation au besoin de retrieval + generation.

### Prompting

Le prompting repose sur deux niveaux :

- un prompt systeme rappelant de ne repondre qu'a partir du contexte ;
- un prompt utilisateur contenant la question et les chunks recuperes.

Contraintes explicites du prompt :

- repondre uniquement a partir du contexte fourni ;
- ne pas inventer d'information absente ;
- citer les elements utiles : titre, ville, date, lieu, conditions ;
- repondre en francais.

### Limites

- dependance a une API distante ;
- cout potentiel des appels embeddings et generation ;
- variabilite possible des resultats selon le modele et le corpus ;
- absence de memoire conversationnelle dans ce POC.

## 5. Construction de la base vectorielle

### FAISS utilise

Le projet utilise le vector store FAISS expose via LangChain avec `FAISS.from_documents(...)`.

Le choix est justifie par :

- une excellente simplicite pour un POC local ;
- une bonne portabilite avec `faiss-cpu` ;
- une integration directe avec LangChain ;
- une performance suffisante sur un corpus de taille moderee.

### Strategie d'indexation

Le pipeline d'indexation est :

1. chargement des documents normalises ;
2. conversion en `Document` LangChain ;
3. chunking ;
4. calcul des embeddings des chunks ;
5. construction de l'index FAISS ;
6. sauvegarde locale.

Le repo ne configure pas d'algorithme FAISS avance de type IVF, HNSW ou PQ. Pour un corpus de quelques centaines d'evenements et un POC local, ce choix est acceptable car il privilegie la fiabilite, la simplicite et la reproductibilite. Si le corpus devait croitre fortement, une strategie ANN plus explicite serait a etudier.

### Persistance

L'index est sauvegarde localement dans `data/faiss/events_index/` avec :

- `index.faiss` : structure vectorielle FAISS ;
- `index.pkl` : donnees serialisees associees au vector store LangChain ;
- `manifest.json` : metadonnees de construction.

Le manifeste conserve notamment :

- nom de l'index ;
- modele d'embedding ;
- `chunk_size` ;
- `chunk_overlap` ;
- nombre de documents sources ;
- nombre de chunks indexes.

### Metadonnees associees

Pour chaque chunk, les metadonnees utiles conservees sont :

- `uid`
- `title`
- `city`
- `location_name`
- `location_address`
- `date`
- `conditions`
- `keywords`
- `canonicalurl`

Ces metadonnees servent a enrichir le contexte et les sources retournees a l'utilisateur.

## 6. API et endpoints exposes

### Framework

L'API est implementee avec FastAPI.

Justification :

- validation automatique via Pydantic ;
- documentation interactive Swagger ;
- code concis et lisible ;
- integration simple avec un service Python deja structure.

### Endpoints

- `GET /health` : verifie que l'API repond.
- `GET /metadata` : expose les parametres metier du corpus actif.
- `POST /ask` : prend une question et retourne une reponse RAG + sources.
- `POST /rebuild` : reconstruit l'index vectoriel apres verification d'un token.

### Format des requetes et reponses

Exemple de requete `POST /ask` :

```json
{
  "question": "Quels evenements gratuits ont lieu a Montpellier ?",
  "top_k": 5
}
```

Exemple de reponse :

```json
{
  "answer": "...",
  "sources": [
    {
      "uid": "123",
      "title": "Nom de l'evenement",
      "city": "Montpellier",
      "date": "2025-09-20T10:00:00+02:00",
      "score": null
    }
  ]
}
```

### Exemples d'appel

```bash
curl -X POST "http://127.0.0.1:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{"question":"Quels evenements musicaux ont lieu a Montpellier ?","top_k":5}'
```

```bash
curl -X POST "http://127.0.0.1:8000/rebuild" \
  -H "Content-Type: application/json" \
  -d '{"token":"change_me_local_token"}'
```

### Gestion des erreurs et limitations

L'API gere notamment :

- questions invalides via validation Pydantic ;
- index absent via erreur 503 ou 404 selon le cas ;
- token de rebuild manquant ou invalide ;
- erreurs inattendues encapsulees en `HTTPException`.

L'endpoint `/rebuild` est protege par un token applicatif. C'est suffisant pour un POC local, mais insuffisant pour une exposition publique reelle. Un durcissement ulterieur devrait ajouter authentification, autorisation et rate limiting.

### Tests API

Le repo contient :

- des tests unitaires / integration sur `/health`, `/metadata`, `/ask` et `/rebuild` ;
- un script `scripts/api_test.py` pour un test fonctionnel rapide ;
- la documentation Swagger disponible sur `/docs`.

## 7. Evaluation du systeme

### Jeu de test annote

Le repo contient un jeu de test annote dans `data/eval/reference_qa.json`, genere automatiquement par le script `scripts/generate_eval_dataset.py` a partir du corpus reel via ChatMistralAI.

Contenu :

- **30 cas** au total ;
- **20 cas positifs** : 2 par categorie (musique, theatre, danse, exposition, gratuit, enfants, sport, patrimoine, conference, festival) ;
- **5 cas negatifs** : questions hors corpus (autre ville, categorie absente) ;
- **5 cas ambigus** : questions dependantes d'une date ou trop vagues.

Schema par entree :

- `id` : identifiant sequentiel ;
- `case_type` : `positive`, `negative` ou `ambiguous` ;
- `question` : formulation en langage naturel ;
- `ground_truth` : reponse factuelle de reference (1-3 phrases) ;
- `expected_keywords` : mots-cles attendus dans la reponse ;
- `expected_city` : ville attendue, `null` pour les cas negatifs et ambigus.

Ce jeu couvre trois registres essentiels pour evaluer un systeme RAG : les cas ou le systeme doit trouver et restituer une information, les cas ou il doit reconnaitre l'absence d'information, et les cas ou la question est trop ouverte pour une reponse deterministe.

### Evaluation heuristique

Le script `evaluate_rag.py` mesure :

- `keyword_coverage` : fraction de mots-cles attendus presents dans la reponse ;
- `city_match` : presence d'au moins une source dans la ville attendue ;
- un label global : `correct`, `partially_correct`, `incorrect`.

Regles appliquees :

- `correct` si `keyword_coverage >= 0.66` et `city_match = true` (cas positifs) ;
- `correct` si signal de refus detecte dans la reponse (cas negatifs : "aucun evenement", "je ne dispose pas", etc.) ;
- `partially_correct` si au moins un critere est partiellement satisfait ;
- `incorrect` sinon.

Resultats sur les 30 cas :

| Indicateur | Valeur |
|---|---|
| Total | 30 |
| Correct | 22 (73 %) |
| Partiellement correct | 3 (10 %) |
| Incorrect | 5 (17 %) |
| Avg keyword coverage | 0.908 |
| Repartition | 20 positifs / 5 negatifs / 5 ambigus |

Interpretation : 73 % de reponses correctes sur un jeu equilibre incluant des cas negatifs est un resultat solide pour un POC. Les 17 % incorrects concernent principalement des cas ou le modele echoue a exprimer un refus explicite sur des questions hors corpus, ou des cas ambigus ou la reponse varie selon la date.

### Evaluation Ragas

#### Metriques retenues et disponibles

RAGAS 0.4.x propose un large catalogue de metriques. Pour ce POC, les metriques retenues sont celles qui :

1. evaluent les trois piliers fondamentaux d'un pipeline RAG (generation, precision, rappel) ;
2. sont compatibles avec un LLM Mistral et des embeddings Mistral (sans dependance a OpenAI ou NVIDIA) ;
3. ont produit des resultats exploitables malgre les contraintes de rate limiting de l'API.

| Metrique | Retenue | Raison |
|---|---|---|
| `faithfulness` | Oui | Mesure si la reponse est ancree dans les documents recuperes (detection d'hallucination). Fondamentale pour evaluer la fidelite RAG. |
| `context_precision` | Oui | Mesure quelle proportion des chunks recuperes est reellement utile a la reponse. Evalue la qualite du retrieval. |
| `context_recall` | Oui | Mesure si les informations necessaires a la reponse sont bien presentes dans les chunks recuperes. Evalue la completude du retrieval. |
| `answer_relevancy` | Non | Necessite de generer des questions synthétiques a partir de la reponse puis de calculer une similarite cosinus. Echecs systematiques sur ce corpus en raison du rate limiting API lors de l'evaluation. Constitue une amelioration future. |
| `context_relevancy` (NV) | Non | Variante specifique NVIDIA (`nv_context_relevance`) dans RAGAS 0.4.x. Incompatible avec les modeles Mistral. Remplacable par `answer_correctness` ou `semantic_similarity` dans une version future. |
| `answer_correctness` | Non retenu | Combine faithfulness et similarite semantique ; necessite un `ground_truth` tres precis et factuel, difficile a garantir sur un dataset genere automatiquement. |
| `semantic_similarity` | Non retenu | Pertinent pour des questions a reponse unique ; peu adapte aux questions ouvertes sur des listes d'evenements. |

#### Resultats obtenus

Script : `scripts/evaluate_ragas.py` sur 30 cas du jeu annote, avec `max_workers=1` pour respecter les limites de l'API Mistral.

| Metrique | Score |
|---|---|
| faithfulness | **0.762** |
| context_precision | **0.575** |
| context_recall | **0.650** |
| answer_relevancy | null (echecs API) |
| context_relevancy | null (metrique NV incompatible) |

#### Interpretation des resultats

**Faithfulness : 0.762**

76 % des reponses generees sont bien ancrees dans les documents recuperes. Ce score indique que le modele respecte generalement la consigne de ne pas inventer d'information absente du contexte. Les 24 % de defaillances concernent principalement les cas negatifs et ambigus, ou le modele a tendance a reformuler des informations approximatives plutot qu'a exprimer un refus explicite.

**Context precision : 0.575**

57,5 % des chunks recuperes par le retriever sont reellement pertinents pour repondre a la question. Ce score moyen revele un bruit notable dans le retrieval : environ 4 documents sur 10 recuperes n'apportent pas d'information utile a la reponse finale. Cela s'explique en partie par l'absence de filtrage par metadonnees (categorie, date) et l'absence de reranker. C'est la metrique la plus directement ameliorable.

**Context recall : 0.650**

65 % des informations factuelles necessaires a la reponse sont presentes dans les chunks recuperes. Un rappel de 0.65 signifie que le retriever manque environ 35 % des informations pertinentes. Plusieurs facteurs l'expliquent : `top_k=5` peut etre insuffisant pour les questions portant sur plusieurs evenements, et certains chunks ne contiennent pas l'information cle en raison du decoupage (chunking).

**Coherence globale**

Les trois scores sont coherents entre eux. Un context recall de 0.65 limite mecaniquement la faithfulness atteignable : si le modele n'a pas acces aux informations pertinentes, il ne peut pas les restituer. La context precision de 0.575 confirme que le retrieval est le maillon le plus fragile du pipeline, ce qui oriente directement les pistes d'amelioration.

### Analyse qualitative

Points positifs observes :

- les reponses mentionnent systematiquement titre, date, lieu et conditions quand les documents les contiennent ;
- le systeme repond de maniere structuree et lisible, en francais ;
- les cas de refus (cas negatifs) sont correctement geres dans 4 cas sur 5.

Limites observees :

- certains chunks recuperes contiennent des evenements generiques (ex. "Mai a Velo 2025") peu specifiques a la question posee ;
- les cas ambigus (questions dependantes de la date) produisent des reponses en general correctes mais pas toujours precises ;
- l'absence de reranker laisse passer des chunks hors sujet dans le top-5.

## 8. Recommandations et perspectives

### Ce qui fonctionne bien

- pipeline RAG complet et relancable ;
- separation claire entre ingestion, indexation, retrieval, generation et API ;
- reconstruction de l'index a la demande ;
- evaluation heuristique et integration Ragas presentes ;
- CI et conteneurisation disponibles.

### Limites du POC

- pas de benchmark de performance formel (latence, debit, memoire) ;
- pas de reranking ;
- pas d'historique conversationnel ;
- dependance a l'API Mistral ;
- deux metriques Ragas non disponibles (answer_relevancy : rate limiting, context_relevancy : metrique NV incompatible avec Mistral) ;
- evaluation limitee a 30 cas generes automatiquement, sans validation humaine exhaustive.

### Ameliorations possibles

- introduire un reranker pour ameliorer la context precision (actuellement 0.575) ;
- ajouter un filtre par seuil de distance FAISS (`max_distance_threshold`) : permettrait d'ecarter les documents trop eloignes semantiquement de la requete, reduisant le bruit dans le retrieval et ameliorant la precision des reponses sur les cas hors corpus ;
- augmenter `top_k` ou implementer un retrieval adaptatif pour ameliorer le context recall (actuellement 0.650) ;
- ajouter des scores de retrieval exploitables dans les sources retournees ;
- completer l'evaluation par de la revue humaine sur un sous-ensemble ;
- ajouter un benchmark de performance simple (latence p50/p95) ;
- etudier un filtrage metadonnees + retrieval hybride (lexical + semantique) ;
- preparer une cible de deploiement plus robuste.

### Passage en production

Pour un deploiement elargi, il faudrait au minimum :

- une authentification forte ;
- une gestion des secrets plus mature ;
- un monitoring ;
- une politique de retry et de timeout mieux instrumentee ;
- un stockage / indexation plus industrialises si le volume augmente.

## 9. Organisation du depot GitHub

Arborescence fonctionnelle :

```text
src/puls_events_rag/
  api/          -> endpoints FastAPI et schemas
  evaluation/   -> evaluation heuristique et Ragas
  ingestion/    -> client OpenDataSoft et preprocessing
  rag/          -> indexer, retriever, prompts, service RAG
  config.py     -> configuration centralisee
  logger.py     -> logs structurees

scripts/
  build_dataset.py
  build_index.py
  evaluate_rag.py
  evaluate_ragas.py
  api_test.py
  run_local.py

data/
  raw/          -> dataset normalise
  eval/         -> jeu annote et resultats d'evaluation
  faiss/        -> index vectoriel persiste

tests/
  unit/
  integration/

docs/
  rapport_technique.md
```

Le depot est organise de facon lisible. Un evaluateur peut identifier rapidement les scripts, les tests, l'API, les donnees et la documentation.

## 10. Annexes

### Extrait du jeu de test annote

```json
{
  "id": "q2",
  "question": "Y a-t-il des evenements gratuits a Montpellier ?",
  "reference_answer": "La reponse doit mentionner des evenements dont les conditions indiquent la gratuite, s'ils existent dans le corpus.",
  "expected_keywords": ["gratuit", "Montpellier"],
  "expected_city": "Montpellier"
}
```

### Prompt systeme

```text
Tu es un assistant specialise dans les evenements culturels publics en France.
Ta mission : repondre uniquement a partir du contexte fourni, rester factuel, clair et utile,
ne jamais inventer d'information absente du contexte.
```

### Exemple de log metier

```text
indexer.raw_documents_loaded
indexer.documents_split
indexer.embeddings_initialized
indexer.faiss_built
retriever.documents_retrieved
rag.answer_generated
```

### Exemple de reponse JSON

```json
{
  "answer": "Voici les evenements gratuits a Montpellier identifies dans le contexte fourni : ...",
  "sources": [
    {
      "uid": "94319571",
      "title": "Conferences, expositions et dedicaces de livres au Cercle Culturel Languedocien !",
      "city": "Montpellier",
      "date": "2025-09-20T10:00:00+02:00",
      "score": null
    }
  ]
}
```

## Conclusion

Le projet livre un POC RAG fonctionnel, relancable et demonstrable, conforme a l'esprit de la mission Puls-Events :

- ingestion de donnees d'evenements ;
- preparation documentaire ;
- chunking ;
- embeddings Mistral ;
- indexation FAISS ;
- retrieval semantique ;
- generation de reponse ;
- API REST ;
- endpoint de reconstruction ;
- tests ;
- evaluation automatisable ;
- Docker et CI.

Le systeme est defendable en soutenance. Les resultats d'evaluation sont honnetes et coherents : 73 % de reponses correctes sur 30 cas equilibres (positifs, negatifs, ambigus), une faithfulness de 0.76 confirmant la fidelite au contexte, et une context precision de 0.575 qui identifie clairement le retrieval comme le maillon a ameliorer. Ces chiffres sont interpretables et orientent directement les pistes d'evolution.
