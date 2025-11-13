# Monitoring et Observabilité

Ce répertoire contient la configuration pour Prometheus et Grafana, permettant de surveiller les performances et la santé du système d'analyse de documents.

## Services

### Prometheus
- **Port**: 9091 (exposé sur l'hôte)
- **URL**: http://localhost:9091
- **Configuration**: `prometheus/prometheus.yml`

Prometheus collecte les métriques depuis :
- L'API FastAPI (port 8000, endpoint `/metrics`)
- Les workers Dramatiq (port 9090, endpoint `/metrics`)

### Grafana
- **Port**: 3000 (exposé sur l'hôte)
- **URL**: http://localhost:3000
- **Identifiants par défaut**:
  - Username: `admin`
  - Password: `admin`

**⚠️ Important**: Changez le mot de passe par défaut en production !

## Métriques Disponibles

### Métriques Custom

1. **`pages_processed_total`** (Counter)
   - Nombre total de pages traitées
   - Labels: `status` (success/failed), `doc_type` (Facture/Attestation_CEE/etc.)

2. **`document_processing_duration_seconds`** (Histogram)
   - Durée de traitement d'un document complet
   - Buckets: 0.1s, 0.5s, 1s, 2s, 5s, 10s, 30s, 60s, 120s, 300s

3. **`page_processing_duration_seconds`** (Histogram)
   - Durée de traitement d'une page individuelle
   - Buckets: 0.1s, 0.5s, 1s, 2s, 5s, 10s, 30s, 60s

4. **`dramatiq_queue_size`** (Gauge)
   - Taille actuelle des files d'attente Dramatiq
   - Labels: `queue_name` (default/ocr-queue/etc.)
   - Mis à jour toutes les 10 secondes par défaut

5. **`documents_processed_total`** (Counter)
   - Nombre total de documents traités
   - Labels: `status` (success/partial_success/error)

6. **`processing_errors_total`** (Counter)
   - Nombre total d'erreurs de traitement
   - Labels: `error_type` (ocr_failed/classification_error/etc.)

### Métriques Standard FastAPI

L'API expose automatiquement les métriques standard via `prometheus-fastapi-instrumentator`:
- Latence des requêtes HTTP
- Nombre de requêtes par endpoint
- Codes de statut HTTP
- Etc.

## Configuration

### Prometheus

La configuration se trouve dans `prometheus/prometheus.yml`. Les principales options :
- `scrape_interval`: Intervalle de collecte des métriques (15s par défaut)
- `storage.tsdb.retention.time`: Durée de rétention des données (30 jours par défaut)

### Grafana

Les dashboards sont automatiquement chargés depuis `grafana/dashboards/`.

Le dashboard par défaut (`doc-analysis-dashboard.json`) affiche :
- Pages traitées par statut et type de document
- Durée de traitement des documents et pages
- Taille des files d'attente Dramatiq
- Erreurs par type
- Documents traités par statut

## Utilisation

### Démarrer les services de monitoring

```bash
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d prometheus grafana
```

### Accéder aux métriques

1. **Prometheus**: http://localhost:9091
   - Utilisez l'onglet "Graph" pour exécuter des requêtes PromQL
   - Exemple: `rate(pages_processed_total[5m])`

2. **Grafana**: http://localhost:3000
   - Connectez-vous avec les identifiants par défaut
   - Le dashboard "Document Analysis - Monitoring Dashboard" devrait être disponible

### Requêtes PromQL utiles

```promql
# Taux de pages traitées par seconde (succès)
rate(pages_processed_total{status="success"}[5m])

# Taux d'erreurs par type
rate(processing_errors_total[5m]) by (error_type)

# P95 de la durée de traitement des documents
histogram_quantile(0.95, rate(document_processing_duration_seconds_bucket[5m]))

# Taille actuelle de la file d'attente
dramatiq_queue_size

# Taux de documents traités par statut
rate(documents_processed_total[5m]) by (status)
```

## Dépannage

### Les métriques ne s'affichent pas dans Prometheus

1. Vérifiez que les services API et workers sont démarrés
2. Vérifiez les logs de Prometheus: `docker logs doc-analysis-prometheus`
3. Vérifiez que les endpoints `/metrics` sont accessibles:
   - API: `curl http://localhost:8000/metrics`
   - Workers: `curl http://localhost:9090/metrics`

### Grafana ne se connecte pas à Prometheus

1. Vérifiez que Prometheus est démarré et accessible
2. Vérifiez la configuration dans `grafana/provisioning/datasources/prometheus.yml`
3. Vérifiez les logs de Grafana: `docker logs doc-analysis-grafana`

## Personnalisation

### Ajouter de nouvelles métriques

1. Ajoutez la métrique dans `src/utils/metrics.py`
2. Utilisez-la dans votre code (workers, API, etc.)
3. Les métriques seront automatiquement exposées sur `/metrics`

### Créer un nouveau dashboard Grafana

1. Créez le dashboard dans l'interface Grafana
2. Exportez-le en JSON
3. Placez-le dans `grafana/dashboards/`
4. Il sera automatiquement chargé au démarrage

## Notes de Production

- **Sécurité**: Changez les mots de passe par défaut de Grafana
- **Rétention**: Ajustez `storage.tsdb.retention.time` selon vos besoins
- **Ressources**: Prometheus peut consommer beaucoup de mémoire avec de grandes quantités de métriques
- **Backup**: Les données de Prometheus sont stockées dans le volume `prometheus_data`

