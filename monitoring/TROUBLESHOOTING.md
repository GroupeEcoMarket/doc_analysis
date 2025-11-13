# Guide de Dépannage - Métriques Prometheus

## Problème : Impossible d'accéder à http://localhost:9090/metrics

### Vérifications à faire

1. **Vérifier l'orthographe de l'URL**
   - ✅ Correct : `http://localhost:9090/metrics` (sans le `/` à la fin)
   - ❌ Incorrect : `http://locahost:9090/metrics/` (faute de frappe + slash final)

2. **Vérifier que le service workers est démarré**
   ```bash
   docker-compose -f docker-compose.yml -f docker-compose.prod.yml ps workers
   ```
   Le service doit être en état "Up".

3. **Vérifier les logs du service workers**
   ```bash
   docker-compose -f docker-compose.yml -f docker-compose.prod.yml logs workers | grep -i metrics
   ```
   Vous devriez voir un message comme :
   ```
   Serveur de métriques démarré sur http://0.0.0.0:9090/metrics
   ```

4. **Vérifier que le port est bien exposé**
   ```bash
   docker-compose -f docker-compose.yml -f docker-compose.prod.yml port workers 9090
   ```
   Devrait afficher : `0.0.0.0:9090`

5. **Tester depuis l'intérieur du conteneur**
   ```bash
   docker-compose -f docker-compose.yml -f docker-compose.prod.yml exec workers curl http://localhost:9090/metrics
   ```
   Si cela fonctionne, le problème vient du mapping de port Docker.

6. **Vérifier que le port n'est pas déjà utilisé sur l'hôte**
   ```bash
   # Sur Linux/Mac
   netstat -tuln | grep 9090
   # ou
   lsof -i :9090
   
   # Sur Windows PowerShell
   netstat -ano | findstr :9090
   ```

### Solutions

#### Solution 1 : Redémarrer le service workers
```bash
docker-compose -f docker-compose.yml -f docker-compose.prod.yml restart workers
```

#### Solution 2 : Vérifier la configuration du port dans docker-compose.prod.yml
Assurez-vous que la section `ports` est présente :
```yaml
workers:
  ports:
    - "9090:9090"
```

#### Solution 3 : Si plusieurs workers sont en cours d'exécution
Avec plusieurs processus workers (DRAMATIQ_PROCESSES > 1), seul le premier processus peut démarrer le serveur sur le port 9090. C'est normal et attendu. Les autres processus afficheront un avertissement dans les logs, mais cela n'empêche pas le fonctionnement.

#### Solution 4 : Vérifier les logs complets
```bash
docker-compose -f docker-compose.yml -f docker-compose.prod.yml logs workers
```
Cherchez les erreurs liées au démarrage du serveur de métriques.

### Test rapide

Pour tester rapidement si le serveur répond :
```bash
# Depuis votre machine
curl http://localhost:9090/metrics

# Ou depuis l'intérieur du conteneur
docker-compose -f docker-compose.yml -f docker-compose.prod.yml exec workers curl http://localhost:9090/metrics
```

### Problèmes courants

1. **"Connection refused"**
   - Le service workers n'est pas démarré
   - Le port n'est pas exposé correctement
   - Le serveur de métriques n'a pas démarré (vérifier les logs)

2. **"DNS_PROBE_FINISHED_NXDOMAIN"**
   - Faute de frappe dans l'URL (ex: "locahost" au lieu de "localhost")
   - Problème de résolution DNS (essayez avec 127.0.0.1)

3. **"Port already in use"**
   - Un autre service utilise le port 9090
   - Plusieurs processus workers tentent d'utiliser le même port (normal, seul le premier réussit)

### Alternative : Accéder via l'API

Si le serveur de métriques des workers ne fonctionne pas, vous pouvez toujours accéder aux métriques de l'API :
```bash
curl http://localhost:8000/metrics
```

Les métriques des workers seront également disponibles via l'API si elles sont partagées dans le même registre Prometheus.

