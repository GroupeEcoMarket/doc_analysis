"""
QA Report Generator
Génère un rapport HTML avec galerie avant/après et statistiques
"""

import os
import json
import statistics
from pathlib import Path
from typing import List, Dict, Optional, Any
from datetime import datetime
import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from src.utils.qa_flags import load_qa_flags, QAFlags
from src.utils.logger import get_logger

logger = get_logger(__name__)


class QAReportGenerator:
    """
    Générateur de rapport QA HTML
    """
    
    def __init__(self, output_dir: str, max_workers: int = 4):
        """
        Initialise le générateur de rapport
        
        Args:
            output_dir: Répertoire contenant les fichiers traités
            max_workers: Nombre de threads pour le traitement parallèle
        """
        self.output_dir = output_dir
        self.pages_data: List[Dict[str, Any]] = []
        self.max_workers = max_workers
        self._image_cache = {}  # Cache pour les images base64
    
    def collect_data(self) -> None:
        """Collecte les données QA de tous les fichiers traités"""
        output_path = Path(self.output_dir)
        
        if not output_path.exists():
            logger.warning(f"Répertoire non trouvé: {self.output_dir}")
            return
        
        # Chercher tous les fichiers .qa.json
        qa_files = list(output_path.rglob("*.qa.json"))
        
        if not qa_files:
            logger.warning(f"Aucun fichier .qa.json trouvé dans {self.output_dir}")
            return
        
        logger.info(f"Trouve {len(qa_files)} fichier(s) .qa.json")
        
        for qa_file in qa_files:
            # Correspondant image file
            image_file = qa_file.with_suffix('').with_suffix('.png')
            if not image_file.exists():
                # Essayer .jpg
                image_file = qa_file.with_suffix('').with_suffix('.jpg')
            
            if not image_file.exists():
                logger.warning(f"Image non trouvée pour {qa_file.name}")
                continue
            
            # Charger les flags QA
            qa_flags = load_qa_flags(str(image_file))
            if qa_flags is None:
                logger.warning(f"Impossible de charger les flags QA pour {image_file.name}")
                continue
            
            # Charger les transformations pour obtenir le masque/contour et les chemins
            transform_file = image_file.with_suffix('.transform.json')
            transform_data = None
            original_image_path = None
            transformed_image_path = str(image_file)
            
            if transform_file.exists():
                try:
                    with open(transform_file, 'r', encoding='utf-8') as f:
                        transform_data = json.load(f)
                    
                    # Utiliser les chemins depuis le fichier transform.json
                    if transform_data:
                        original_image_path = transform_data.get('output_original_path')
                        # output_path est l'image transformée
                        transformed_from_json = transform_data.get('output_path')
                        if transformed_from_json and os.path.exists(transformed_from_json):
                            transformed_image_path = transformed_from_json
                        # Fallback : utiliser output_transformed_path si présent (ancien format)
                        elif not transformed_from_json:
                            transformed_from_json = transform_data.get('output_transformed_path')
                            if transformed_from_json and os.path.exists(transformed_from_json):
                                transformed_image_path = transformed_from_json
                except Exception as e:
                    logger.warning(f"Erreur lors du chargement des transformations pour {image_file.name}", exc_info=True)
            
            # Fallback : chercher l'image originale si pas dans transform.json
            if not original_image_path or not os.path.exists(original_image_path):
                if '_transformed' in image_file.stem:
                    original_image_path = str(image_file).replace('_transformed', '_original')
                else:
                    original_image_path = self._find_original_image(image_file)
            
            page_data = {
                'page_name': image_file.stem,
                'input_path': original_image_path if original_image_path and os.path.exists(original_image_path) else None,
                'output_path': transformed_image_path,
                'qa_flags': qa_flags.to_dict(),
                'transform_data': transform_data,
                'page_index': len(self.pages_data)  # Ajouter l'index pour le lazy loading
            }
            self.pages_data.append(page_data)
    
    def _find_input_file(self, output_file: Path) -> Optional[Path]:
        """Trouve le fichier d'entrée correspondant (déprécié, utiliser _find_original_image)"""
        return self._find_original_image(output_file)
    
    def _find_original_image(self, output_file: Path) -> Optional[str]:
        """Trouve l'image originale (_original.png) correspondante"""
        # Chercher l'image _original dans le même répertoire
        if '_transformed' in output_file.stem:
            original_path = str(output_file).replace('_transformed', '_original')
            if os.path.exists(original_path):
                return original_path
        
        # Chercher dans le même répertoire avec différents patterns
        output_dir = output_file.parent
        base_name = output_file.stem.replace('_transformed', '').replace('_page1', '').replace('_page2', '').replace('_page3', '').replace('_page4', '')
        
        # Essayer plusieurs patterns
        patterns = [
            f"{base_name}_original.png",
            f"{base_name}_page1_original.png",
            f"{base_name}_page2_original.png",
        ]
        
        for pattern in patterns:
            candidate = output_dir / pattern
            if candidate.exists():
                return str(candidate)
        
        return None
    
    def generate_html_report(self, output_path: str = "qa_report.html") -> None:
        """
        Génère le rapport HTML
        
        Args:
            output_path: Chemin du fichier HTML de sortie
        """
        if not self.pages_data:
            self.collect_data()
        
        html_content = self._generate_html_content()
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"Rapport QA genere: {output_path}")
    
    def _generate_html_content(self) -> str:
        """Génère le contenu HTML du rapport"""
        
        # Calculer les statistiques (toujours retourne un dict complet)
        stats = self._calculate_statistics()
        
        # Générer les vignettes
        gallery_html = self._generate_gallery() if self.pages_data else '<p>Aucune page à afficher</p>'
        
        # Générer le tableau des flags
        flags_table = self._generate_flags_table() if self.pages_data else '<p>Aucune donnée disponible</p>'
        
        html = f"""<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Rapport QA - Document Analysis Pipeline</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }}
        .header {{
            background-color: #2c3e50;
            color: white;
            padding: 20px;
            border-radius: 5px;
            margin-bottom: 20px;
        }}
        .stats {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-bottom: 30px;
        }}
        .stat-card {{
            background: white;
            padding: 15px;
            border-radius: 5px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .stat-card h3 {{
            margin: 0 0 10px 0;
            color: #2c3e50;
            font-size: 14px;
        }}
        .stat-value {{
            font-size: 24px;
            font-weight: bold;
            color: #3498db;
        }}
        .flags-table {{
            width: 100%;
            background: white;
            border-collapse: collapse;
            margin-bottom: 30px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .flags-table th {{
            background-color: #34495e;
            color: white;
            padding: 12px;
            text-align: left;
            position: relative;
        }}
        .flags-table th.sortable {{
            cursor: pointer;
            user-select: none;
        }}
        .flags-table th.sortable:hover {{
            background-color: #2c3e50;
        }}
        .flags-table th.sortable.sorted-asc .sort-arrow {{
            color: #3498db;
        }}
        .flags-table th.sortable.sorted-asc .sort-arrow::after {{
            content: ' ▲';
        }}
        .flags-table th.sortable.sorted-desc .sort-arrow {{
            color: #3498db;
        }}
        .flags-table th.sortable.sorted-desc .sort-arrow::after {{
            content: ' ▼';
        }}
        .sort-arrow {{
            font-size: 0.8em;
            opacity: 0.5;
        }}
        .flags-table td {{
            padding: 10px;
            border-bottom: 1px solid #ddd;
        }}
        .flags-table tr:hover {{
            background-color: #f5f5f5;
        }}
        .flag-active {{
            color: #e74c3c;
            font-weight: bold;
        }}
        .flag-inactive {{
            color: #95a5a6;
        }}
        .flag-info {{
            color: #2980b9;
            font-weight: bold;
        }}
        .flag-na {{
            color: #bdc3c7;
        }}
        .gallery {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(600px, 1fr));
            gap: 30px;
            margin-bottom: 30px;
        }}
        .gallery-item {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.15);
        }}
        .gallery-item h4 {{
            margin-top: 0;
            margin-bottom: 15px;
            color: #2c3e50;
            font-size: 16px;
            border-bottom: 2px solid #3498db;
            padding-bottom: 8px;
        }}
        .gallery-images {{
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 15px;
            margin-bottom: 15px;
        }}
        .gallery-images > div {{
            text-align: center;
        }}
        .gallery-images strong {{
            display: block;
            margin-bottom: 8px;
            color: #34495e;
            font-size: 13px;
        }}
        .gallery-images img {{
            width: 100%;
            height: auto;
            border: 2px solid #ddd;
            border-radius: 4px;
            cursor: pointer;
            transition: transform 0.2s, box-shadow 0.2s;
        }}
        .gallery-images img:hover {{
            transform: scale(1.05);
            box-shadow: 0 4px 12px rgba(0,0,0,0.2);
            border-color: #3498db;
        }}
        /* Modal pour agrandir les images */
        .modal {{
            display: none;
            position: fixed;
            z-index: 1000;
            left: 0;
            top: 0;
            width: 100%;
            height: 100%;
            background-color: rgba(0,0,0,0.9);
            overflow: auto;
        }}
        .modal-content {{
            margin: 2% auto;
            display: block;
            max-width: 95%;
            max-height: 95%;
            object-fit: contain;
        }}
        .modal-close {{
            position: absolute;
            top: 20px;
            right: 40px;
            color: #f1f1f1;
            font-size: 40px;
            font-weight: bold;
            cursor: pointer;
            transition: 0.3s;
        }}
        .modal-close:hover {{
            color: #bbb;
        }}
        .modal-caption {{
            text-align: center;
            color: #ccc;
            padding: 10px;
            font-size: 18px;
        }}
        /* Mode comparaison côte à côte */
        .modal-compare {{
            display: flex;
            justify-content: center;
            align-items: flex-start;
            gap: 10px;
            padding: 20px;
            max-width: 98%;
            margin: 0 auto;
        }}
        .modal-compare > div {{
            flex: 1;
            text-align: center;
        }}
        .modal-compare img {{
            width: 100%;
            height: auto;
            max-height: 90vh;
            object-fit: contain;
            border: 2px solid #3498db;
            border-radius: 4px;
        }}
        .modal-compare .image-label {{
            color: #3498db;
            font-weight: bold;
            margin-bottom: 10px;
            font-size: 16px;
        }}
        .compare-button {{
            display: inline-block;
            margin-top: 10px;
            padding: 8px 16px;
            background-color: #3498db;
            color: white;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-size: 14px;
            font-weight: bold;
            transition: background-color 0.2s;
        }}
        .compare-button:hover {{
            background-color: #2980b9;
        }}
        .flags-list {{
            margin-top: 10px;
        }}
        .flag-badge {{
            display: inline-block;
            padding: 3px 8px;
            margin: 2px;
            border-radius: 3px;
            font-size: 11px;
            cursor: help;
            font-weight: 500;
        }}
        /* Vert : Action appliquée avec succès */
        .flag-badge.badge-applied {{
            background-color: #27ae60;
            color: white;
        }}
        /* Orange : Action non appliquée à cause des seuils/config */
        .flag-badge.badge-skipped {{
            background-color: #f39c12;
            color: white;
        }}
        /* Bleu : Information/indication */
        .flag-badge.badge-info {{
            background-color: #3498db;
            color: white;
        }}
        /* Rouge : Problème/avertissement détecté */
        .flag-badge.badge-warning {{
            background-color: #e74c3c;
            color: white;
        }}
        /* Gris : Non applicable/non actif */
        .flag-badge.badge-inactive {{
            background-color: #ecf0f1;
            color: #7f8c8d;
        }}
        /* Anciens styles pour compatibilité */
        .flag-badge.active {{
            background-color: #e74c3c;
            color: white;
        }}
        .flag-badge.inactive {{
            background-color: #ecf0f1;
            color: #7f8c8d;
        }}
        
        /* Animation pour le loader */
        @keyframes spin {{
            0% {{ transform: rotate(0deg); }}
            100% {{ transform: rotate(360deg); }}
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>📊 Rapport QA - Document Analysis Pipeline</h1>
        <p>Généré le {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        <p>Total de pages analysées: {len(self.pages_data)}</p>
    </div>
    
    <div class="stats">
        <div class="stat-card">
            <h3>📄 Type de Capture</h3>
            <div class="stat-value" style="color: #3498db;">
                SCAN: {stats['scan_count']} | PHOTO: {stats['photo_count']}
            </div>
        </div>
        <div class="stat-card">
            <h3>✅ Rotations Appliquées</h3>
            <div class="stat-value" style="color: #27ae60;">{stats['rotations_applied']} / {len(self.pages_data)}</div>
        </div>
        <div class="stat-card">
            <h3>✅ Deskews Appliqués</h3>
            <div class="stat-value" style="color: #27ae60;">{stats['deskews_applied']} / {len(self.pages_data)}</div>
        </div>
        <div class="stat-card">
            <h3>✅ Crops Appliqués</h3>
            <div class="stat-value" style="color: #27ae60;">{stats['crops_applied']} / {len(self.pages_data)}</div>
        </div>
        <div class="stat-card">
            <h3>⚠️ Pages avec Avertissements</h3>
            <div class="stat-value" style="color: #e74c3c;">{stats['pages_with_warnings']} / {len(self.pages_data)}</div>
        </div>
    </div>
    
    <!-- Section dédiée aux statistiques de temps de traitement -->
    <h2>⏱️ Statistiques de Temps de Traitement</h2>
    {self._get_time_distribution_warning(stats)}
    <div class="stats" style="margin-bottom: 30px;">
        <div class="stat-card">
            <h3>📊 Temps Moyen</h3>
            <div class="stat-value" style="color: #7f8c8d; font-size: 28px;">{stats['avg_time']:.2f}s</div>
            <div style="font-size: 12px; color: #95a5a6; margin-top: 5px;">Moyenne arithmétique</div>
        </div>
        <div class="stat-card">
            <h3>📈 Temps Médian</h3>
            <div class="stat-value" style="color: #3498db; font-size: 28px;">{stats['median_time']:.2f}s</div>
            <div style="font-size: 12px; color: #95a5a6; margin-top: 5px;">Valeur médiane (50%)</div>
        </div>
        <div class="stat-card">
            <h3>⚡ Temps Minimum</h3>
            <div class="stat-value" style="color: #27ae60; font-size: 28px;">{stats['min_time']:.2f}s</div>
            <div style="font-size: 12px; color: #95a5a6; margin-top: 5px;">Traitement le plus rapide</div>
        </div>
        <div class="stat-card">
            <h3>🐌 Temps Maximum</h3>
            <div class="stat-value" style="color: #e74c3c; font-size: 28px;">{stats['max_time']:.2f}s</div>
            <div style="font-size: 12px; color: #95a5a6; margin-top: 5px;">Traitement le plus lent</div>
        </div>
        <div class="stat-card">
            <h3>📉 Écart Moyen-Médian</h3>
            <div class="stat-value" style="color: #f39c12; font-size: 28px;">{abs(stats['avg_time'] - stats['median_time']):.2f}s</div>
            <div style="font-size: 12px; color: #95a5a6; margin-top: 5px;">Différence moyenne/médiane</div>
        </div>
        <div class="stat-card">
            <h3>📏 Plage (Max - Min)</h3>
            <div class="stat-value" style="color: #9b59b6; font-size: 28px;">{stats['max_time'] - stats['min_time']:.2f}s</div>
            <div style="font-size: 12px; color: #95a5a6; margin-top: 5px;">Écart entre min et max</div>
        </div>
    </div>
    
    <h2>Tableau des Flags par Page</h2>
    {flags_table}
    
    <h2>Galerie Avant/Après</h2>
    <p style="color: #7f8c8d; font-style: italic;">💡 Cliquez sur une image pour l'agrandir</p>
    {gallery_html}

    <!-- Modal pour agrandir les images -->
    <div id="imageModal" class="modal">
        <span class="modal-close">&times;</span>
        <img class="modal-content" id="modalImage">
        <div class="modal-caption" id="modalCaption"></div>
    </div>

    <script>
        (function() {{
            const table = document.querySelector('.flags-table');
            if (!table) return;

            const headers = table.querySelectorAll('th.sortable');
            if (!headers.length) return;

            // Fonction pour parser les valeurs des cellules
            function parseCellValue(cell) {{
                const text = cell.textContent.trim();
                
                // Gérer les temps (ex: "1.23s")
                if (text.endsWith('s')) {{
                    const num = parseFloat(text.slice(0, -1).replace(',', '.'));
                    return isNaN(num) ? text : num;
                }}
                
                // Gérer les angles (ex: "90°")
                if (text.includes('°')) {{
                    const num = parseFloat(text.replace('°', '').replace(',', '.'));
                    return isNaN(num) ? text : num;
                }}
                
                // Gérer les symboles (⚠️, —, 🔄, etc.)
                if (text === '⚠️') return 1;
                if (text === '—') return 0;
                if (text === '🔄') return 1;
                if (text === 'Oui') return 1;
                if (text === 'Non') return 0;
                if (text === 'N/A') return -1;
                
                // Essayer de parser comme nombre
                const numeric = parseFloat(text.replace(/[^0-9\-\.]/g, ''));
                if (!isNaN(numeric)) {{
                    return numeric;
                }}
                
                return text.toLowerCase();
            }}

            // Fonction de tri
            function sortTable(columnIndex, order) {{
                const tbody = table.querySelector('tbody');
                const rows = Array.from(tbody.querySelectorAll('tr'));

                rows.sort((rowA, rowB) => {{
                    const cellA = rowA.children[columnIndex];
                    const cellB = rowB.children[columnIndex];
                    const valueA = parseCellValue(cellA);
                    const valueB = parseCellValue(cellB);

                    let comparison = 0;
                    if (typeof valueA === 'number' && typeof valueB === 'number') {{
                        comparison = valueA - valueB;
                    }} else {{
                        comparison = valueA.toString().localeCompare(valueB.toString(), 'fr', {{ numeric: true }});
                    }}

                    return order === 'asc' ? comparison : -comparison;
                }});

                // Réinsérer les lignes triées
                rows.forEach(row => tbody.appendChild(row));
            }}

            // Ajouter les événements de clic sur les en-têtes
            headers.forEach(header => {{
                header.addEventListener('click', () => {{
                    const columnIndex = parseInt(header.dataset.column, 10);
                    
                    // Déterminer l'ordre actuel
                    let currentOrder = 'none';
                    if (header.classList.contains('sorted-asc')) {{
                        currentOrder = 'asc';
                    }} else if (header.classList.contains('sorted-desc')) {{
                        currentOrder = 'desc';
                    }}
                    
                    // Calculer le nouvel ordre
                    let newOrder = 'asc';
                    if (currentOrder === 'asc') {{
                        newOrder = 'desc';
                    }} else if (currentOrder === 'desc') {{
                        newOrder = 'asc';
                    }}
                    
                    // Retirer les classes de tri de tous les en-têtes
                    headers.forEach(h => {{
                        h.classList.remove('sorted-asc', 'sorted-desc');
                    }});
                    
                    // Ajouter la classe de tri à l'en-tête actuel
                    header.classList.add('sorted-' + newOrder);
                    
                    // Trier le tableau
                    sortTable(columnIndex, newOrder);
                }});
            }});
        }})();

        // Modal pour agrandir les images
        (function() {{
            const modal = document.getElementById('imageModal');
            const modalImg = document.getElementById('modalImage');
            const modalCaption = document.getElementById('modalCaption');
            const closeBtn = document.querySelector('.modal-close');
            
            if (!modal || !modalImg || !closeBtn) return;
            
            // Ajouter un événement de clic à toutes les images de la galerie
            const galleryImages = document.querySelectorAll('.gallery-images img');
            galleryImages.forEach(img => {{
                img.addEventListener('click', function() {{
                    modal.style.display = 'block';
                    modalImg.src = this.src;
                    
                    // Récupérer le titre de la page et le type d'image
                    const galleryItem = this.closest('.gallery-item');
                    const pageTitle = galleryItem ? galleryItem.querySelector('h4').textContent : '';
                    const imageType = this.alt || '';
                    modalCaption.textContent = `${{pageTitle}} - ${{imageType}}`;
                }});
            }});
            
            // Fermer le modal en cliquant sur le X
            closeBtn.addEventListener('click', function() {{
                modal.style.display = 'none';
            }});
            
            // Fermer le modal en cliquant en dehors de l'image
            modal.addEventListener('click', function(event) {{
                if (event.target === modal) {{
                    modal.style.display = 'none';
                }}
            }});
            
            // Fermer le modal avec la touche Échap
            document.addEventListener('keydown', function(event) {{
                if (event.key === 'Escape' && modal.style.display === 'block') {{
                    modal.style.display = 'none';
                }}
            }});
        }})();

        // Fonction globale pour ouvrir le mode comparaison
        window.openCompareModal = function(button) {{
            const modal = document.getElementById('imageModal');
            const modalImg = document.getElementById('modalImage');
            const modalCaption = document.getElementById('modalCaption');
            
            if (!modal) return;
            
            // Récupérer les données de la carte
            const galleryItem = button.closest('.gallery-item');
            const pageTitle = galleryItem.querySelector('h4').textContent;
            const sourceData = galleryItem.getAttribute('data-source');
            const maskData = galleryItem.getAttribute('data-mask');
            const outputData = galleryItem.getAttribute('data-output');
            
            // Créer le contenu de comparaison
            let compareHTML = '<div class="modal-compare">';
            
            if (sourceData) {{
                compareHTML += `
                    <div>
                        <div class="image-label">Source</div>
                        <img src="data:image/jpeg;base64,${{sourceData}}" alt="Source">
                    </div>
                `;
            }}
            
            if (maskData) {{
                compareHTML += `
                    <div>
                        <div class="image-label">Masque/Contour</div>
                        <img src="data:image/jpeg;base64,${{maskData}}" alt="Masque">
                    </div>
                `;
            }}
            
            if (outputData) {{
                compareHTML += `
                    <div>
                        <div class="image-label">Final</div>
                        <img src="data:image/jpeg;base64,${{outputData}}" alt="Final">
                    </div>
                `;
            }}
            
            compareHTML += '</div>';
            
            // Remplacer le contenu du modal
            modalImg.style.display = 'none';
            modalImg.insertAdjacentHTML('afterend', compareHTML);
            modalCaption.textContent = `${{pageTitle}} - Comparaison`;
            
            modal.style.display = 'block';
            
            // Nettoyer au fermeture
            const closeModal = function() {{
                const compareDiv = modal.querySelector('.modal-compare');
                if (compareDiv) {{
                    compareDiv.remove();
                }}
                modalImg.style.display = 'block';
                modal.style.display = 'none';
            }};
            
            // Réattacher les événements de fermeture
            const closeBtn = modal.querySelector('.modal-close');
            closeBtn.onclick = closeModal;
            modal.onclick = function(event) {{
                if (event.target === modal) {{
                    closeModal();
                }}
            }};
        }};

        // ============================================
        // LAZY LOADING / SCROLL INFINI POUR LA GALERIE
        // ============================================
        document.addEventListener('DOMContentLoaded', function() {{
            const galleryContainer = document.getElementById('galleryContainer');
            const galleryLoader = document.getElementById('galleryLoader');
            const galleryEnd = document.getElementById('galleryEnd');
            const galleryDataScript = document.getElementById('galleryItemsData');
            
            if (!galleryContainer || !galleryLoader || !galleryEnd || !galleryDataScript) return;
            
            // Charger les données pré-générées
            let allGalleryItems = [];
            try {{
                allGalleryItems = JSON.parse(galleryDataScript.textContent);
                console.log(`Données de galerie chargées: ${{allGalleryItems.length}} items`);
            }} catch (e) {{
                console.error('Erreur lors du chargement des données de galerie:', e);
                return;
            }}
            
            let currentIndex = 10;  // On a déjà chargé les 10 premières
            const batchSize = 10;  // Charger 10 pages à la fois
            let isLoading = false;
            
            // Fonction pour charger un batch de pages
            function loadMorePages() {{
                if (isLoading || currentIndex >= allGalleryItems.length) {{
                    if (currentIndex >= allGalleryItems.length) {{
                        galleryLoader.style.display = 'none';
                        galleryEnd.style.display = 'block';
                    }}
                    return;
                }}
                
                isLoading = true;
                galleryLoader.style.display = 'block';
                
                // Charger le batch avec un petit délai pour l'effet visuel
                setTimeout(() => {{
                    const endIndex = Math.min(currentIndex + batchSize, allGalleryItems.length);
                    const itemsToLoad = allGalleryItems.slice(currentIndex, endIndex);
                    
                    // Ajouter les items HTML au conteneur
                    itemsToLoad.forEach(itemHTML => {{
                        if (itemHTML) {{
                            galleryContainer.insertAdjacentHTML('beforeend', itemHTML);
                        }}
                    }});
                    
                    // Réattacher les événements pour les nouvelles images
                    attachImageEvents();
                    attachCompareEvents();
                    
                    currentIndex = endIndex;
                    isLoading = false;
                    
                    // Vérifier si on a tout chargé
                    if (currentIndex >= allGalleryItems.length) {{
                        galleryLoader.style.display = 'none';
                        galleryEnd.style.display = 'block';
                    }} else {{
                        // Garder le loader visible pour continuer à observer
                        galleryLoader.style.display = 'block';
                    }}
                    
                    console.log(`Charge pages ${{currentIndex - itemsToLoad.length + 1}} a ${{currentIndex}} (${{currentIndex}}/${{allGalleryItems.length}})`);
                }}, 300);
            }}
            
            // Fonction pour attacher les événements aux images (modal)
            function attachImageEvents() {{
                const modal = document.getElementById('imageModal');
                const modalImg = document.getElementById('modalImage');
                const modalCaption = document.getElementById('modalCaption');
                const closeBtn = modal.querySelector('.modal-close');
                
                document.querySelectorAll('.gallery-images img').forEach(img => {{
                    if (!img.dataset.eventAttached) {{
                        img.onclick = function() {{
                            modal.style.display = 'block';
                            modalImg.src = this.src;
                            const pageTitle = this.closest('.gallery-item').querySelector('h4').textContent;
                            const imageType = this.alt || 'Image';
                            modalCaption.textContent = `${{pageTitle}} - ${{imageType}}`;
                        }};
                        img.dataset.eventAttached = 'true';
                    }}
                }});
            }}
            
            // Fonction pour attacher les événements aux boutons de comparaison
            function attachCompareEvents() {{
                document.querySelectorAll('.compare-btn').forEach(btn => {{
                    if (!btn.dataset.eventAttached) {{
                        btn.onclick = function() {{
                            const galleryItem = this.closest('.gallery-item');
                            const pageTitle = galleryItem.querySelector('h4').textContent;
                            window.openCompareModal(galleryItem, pageTitle);
                        }};
                        btn.dataset.eventAttached = 'true';
                    }}
                }});
            }}
            
            // Détection du scroll avec Intersection Observer
            const observer = new IntersectionObserver((entries) => {{
                entries.forEach(entry => {{
                    if (entry.isIntersecting && !isLoading) {{
                        loadMorePages();
                    }}
                }});
            }}, {{
                rootMargin: '300px'  // Commencer à charger 300px avant d'atteindre le loader
            }});
            
            // Observer le loader
            observer.observe(galleryLoader);
            
            // Afficher le loader si on a plus de pages à charger
            if (currentIndex < allGalleryItems.length) {{
                galleryLoader.style.display = 'block';
            }} else {{
                galleryEnd.style.display = 'block';
            }}
            
            console.log(`Galerie initialisee: ${{currentIndex}}/${{allGalleryItems.length}} pages chargees`);
            
            // Debug: vérifier que le loader est visible
            console.log(`Loader visible: ${{galleryLoader.style.display}}`);
            console.log(`Position du loader:`, galleryLoader.getBoundingClientRect());
        }});
    </script>
</body>
</html>
"""
        return html
    
    def _calculate_statistics(self) -> Dict[str, Any]:
        """Calcule les statistiques agrégées"""
        # Valeurs par défaut
        default_stats = {
            'orientation_accuracy': 0.0,
            'overcrop_risk_count': 0,
            'no_quad_rate': 0.0,
            'avg_time': 0.0,
            'median_time': 0.0,
            'min_time': 0.0,
            'max_time': 0.0,
            'avg_deskew_angle': 0.0,
            'avg_deskew_confidence': 0.0,
            'pages_with_flags': 0
        }
        
        if not self.pages_data:
            return default_stats
        
        total = len(self.pages_data)
        
        # Orientation accuracy (pages sans low_confidence_orientation)
        orientation_ok = sum(1 for p in self.pages_data 
                           if not p['qa_flags'].get('low_confidence_orientation', False))
        orientation_accuracy = orientation_ok / total if total > 0 else 0.0
        
        # Overcrop risk
        overcrop_risk_count = sum(1 for p in self.pages_data 
                                 if p['qa_flags'].get('overcrop_risk', False))
        
        # No quad detected
        no_quad_count = sum(1 for p in self.pages_data 
                           if p['qa_flags'].get('no_quad_detected', False))
        no_quad_rate = no_quad_count / total if total > 0 else 0.0
        
        # Statistiques de temps de traitement (distribution complète)
        # Collecter les temps depuis qa_flags (source principale)
        times = []
        for p in self.pages_data:
            processing_time = p['qa_flags'].get('processing_time', 0)
            if processing_time > 0:
                times.append(processing_time)
        
        if times:
            avg_time = statistics.mean(times)
            median_time = statistics.median(times)
            min_time = min(times)
            max_time = max(times)
        else:
            avg_time = 0.0
            median_time = 0.0
            min_time = 0.0
            max_time = 0.0

        # Deskew stats
        deskew_angles = [abs(p['qa_flags'].get('deskew_angle', 0.0))
                         for p in self.pages_data]
        deskew_confidences = [p['qa_flags'].get('deskew_confidence', 0.0)
                              for p in self.pages_data if p['qa_flags'].get('deskew_confidence', 0.0) > 0]
        avg_deskew_angle = statistics.mean(deskew_angles) if deskew_angles else 0.0
        avg_deskew_confidence = statistics.mean(deskew_confidences) if deskew_confidences else 0.0
        
        # Capture type stats
        scan_count = sum(1 for p in self.pages_data 
                        if p['qa_flags'].get('capture_type', 'UNKNOWN') == 'SCAN')
        photo_count = sum(1 for p in self.pages_data 
                         if p['qa_flags'].get('capture_type', 'UNKNOWN') == 'PHOTO')
        
        # Statistiques des transformations appliquées
        rotations_applied = sum(1 for p in self.pages_data if p['qa_flags'].get('rotated', False))
        deskews_applied = sum(1 for p in self.pages_data if abs(p['qa_flags'].get('deskew_angle', 0.0)) >= 0.1)
        crops_applied = sum(1 for p in self.pages_data if p['qa_flags'].get('dewarp_applied', False))
        
        # Pages avec flags d'avertissement
        pages_with_warnings = sum(1 for p in self.pages_data 
                              if any([
                                  p['qa_flags'].get('low_confidence_orientation', False),
                                  p['qa_flags'].get('overcrop_risk', False),
                                  p['qa_flags'].get('no_quad_detected', False),
                                  p['qa_flags'].get('low_contrast_after_enhance', False),
                                  p['qa_flags'].get('too_small_final', False)
                              ]))
        
        return {
            'orientation_accuracy': orientation_accuracy,
            'overcrop_risk_count': overcrop_risk_count,
            'no_quad_rate': no_quad_rate,
            'avg_time': avg_time,
            'median_time': median_time,
            'min_time': min_time,
            'max_time': max_time,
            'avg_deskew_angle': avg_deskew_angle,
            'avg_deskew_confidence': avg_deskew_confidence,
            'pages_with_warnings': pages_with_warnings,
            'scan_count': scan_count,
            'photo_count': photo_count,
            'rotations_applied': rotations_applied,
            'deskews_applied': deskews_applied,
            'crops_applied': crops_applied
        }
    
    def _get_time_distribution_warning(self, stats: Dict[str, Any]) -> str:
        """Génère un avertissement si la moyenne est très différente de la médiane"""
        if stats['median_time'] > 0:
            diff = abs(stats['avg_time'] - stats['median_time'])
            diff_percent = (diff / stats['median_time']) * 100 if stats['median_time'] > 0 else 0
            
            if diff_percent > 30:  # Si l'écart est > 30% de la médiane
                warning_type = 'Valeurs aberrantes detectees' if diff_percent > 50 else 'Distribution asymetrique'
                return f'''
                <p style="margin: 0 0 10px 0; font-size: 11px; color: #f39c12; font-style: italic;">
                    Ecart moyen/mediane: {diff:.2f}s ({diff_percent:.0f}%) - {warning_type}
                </p>
                '''
        return ''
    
    def _generate_flags_table(self) -> str:
        """Génère le tableau HTML des flags"""
        if not self.pages_data:
            return '<p>Aucune donnée disponible</p>'
        
        total_pages = len(self.pages_data)
        
        rows = []
        for page in self.pages_data:
            flags = page['qa_flags']
            
            # Récupérer l'angle de rotation ONNX
            rotation_angle = flags.get('orientation_angle', 0)
            rotation_display = f"{rotation_angle}°" if rotation_angle != 0 else "0°"
            
            # Récupérer la confiance d'orientation
            orientation_conf = flags.get('orientation_confidence', 0.0)
            conf_display = f"{orientation_conf:.2f}" if orientation_conf > 0 else "N/A"
            conf_color = "#27ae60" if orientation_conf >= 0.70 else "#e74c3c" if orientation_conf > 0 else "#7f8c8d"

            # Informations deskew
            deskew_angle = float(flags.get('deskew_angle', 0.0))
            deskew_conf = float(flags.get('deskew_confidence', 0.0))
            deskew_angle_display = f"{deskew_angle:.2f}°" if abs(deskew_angle) > 0 else "0°"
            if deskew_conf > 0:
                deskew_conf_display = f"{deskew_conf:.2f}"
                if deskew_conf >= 0.75:
                    deskew_conf_color = "#27ae60"
                elif deskew_conf >= 0.4:
                    deskew_conf_color = "#f39c12"
                else:
                    deskew_conf_color = "#d35400"
            else:
                deskew_conf_display = "N/A"
                deskew_conf_color = "#7f8c8d"

            # Flags booléens
            def flag_cell(flag_value, true_icon='⚠️', false_icon='—'):
                return (
                    true_icon if flag_value else false_icon,
                    'flag-active' if flag_value else 'flag-inactive'
                )

            low_conf_icon, low_conf_class = flag_cell(flags.get('low_confidence_orientation', False))
            overcrop_icon, overcrop_class = flag_cell(flags.get('overcrop_risk', False))
            no_quad_icon, no_quad_class = flag_cell(flags.get('no_quad_detected', False))
            dewarp_icon, dewarp_class = flag_cell(flags.get('dewarp_applied', False))
            low_contrast_icon, low_contrast_class = flag_cell(flags.get('low_contrast_after_enhance', False))
            too_small_icon, too_small_class = flag_cell(flags.get('too_small_final', False))
            if flags.get('rotated', False):
                rotated_icon = 'Oui'
                rotated_class = 'flag-info'
            else:
                rotated_icon = 'Non'
                rotated_class = 'flag-na'
            
            page_row = f"""
            <tr>
                <td>{page['page_name']}</td>
                <td style="text-align: center; font-weight: bold;">
                    {rotation_display}
                </td>
                <td style="text-align: center; color: {conf_color}; font-weight: bold;">
                    {conf_display}
                </td>
                <td style="text-align: center; font-weight: bold;">
                    {deskew_angle_display}
                </td>
                <td style="text-align: center; color: {deskew_conf_color}; font-weight: bold;">
                    {deskew_conf_display}
                </td>
                <td class="{low_conf_class}" style="text-align: center;">
                    {low_conf_icon}
                </td>
                <td class="{overcrop_class}" style="text-align: center;">
                    {overcrop_icon}
                </td>
                <td class="{no_quad_class}" style="text-align: center;">
                    {no_quad_icon}
                </td>
                <td class="{dewarp_class}" style="text-align: center;">
                    {dewarp_icon}
                </td>
                <td class="{low_contrast_class}" style="text-align: center;">
                    {low_contrast_icon}
                </td>
                <td class="{too_small_class}" style="text-align: center;">
                    {too_small_icon}
                </td>
                <td class="{rotated_class}" style="text-align: center;">
                    {rotated_icon}
                </td>
                <td style="text-align: center; font-weight: bold;">
                    {flags.get('capture_type', 'UNKNOWN')}
                </td>
                <td>{flags.get('processing_time', 0):.2f}s</td>
            </tr>
            """
            rows.append(page_row)
        
        return f"""
        <table class="flags-table">
            <thead>
                <tr>
                    <th class="sortable" data-column="0">Page <span class="sort-arrow">⇅</span></th>
                    <th class="sortable" data-column="1">Orientation (deg) <span class="sort-arrow">⇅</span></th>
                    <th class="sortable" data-column="2">Conf. Orientation <span class="sort-arrow">⇅</span></th>
                    <th class="sortable" data-column="3">Deskew (deg) <span class="sort-arrow">⇅</span></th>
                    <th class="sortable" data-column="4">Conf. Deskew <span class="sort-arrow">⇅</span></th>
                    <th class="sortable" data-column="5">Low Conf <span class="sort-arrow">⇅</span></th>
                    <th class="sortable" data-column="6">Overcrop <span class="sort-arrow">⇅</span></th>
                    <th class="sortable" data-column="7">No Quad <span class="sort-arrow">⇅</span></th>
                    <th class="sortable" data-column="8">Dewarp <span class="sort-arrow">⇅</span></th>
                    <th class="sortable" data-column="9">Low Contrast <span class="sort-arrow">⇅</span></th>
                    <th class="sortable" data-column="10">Too Small <span class="sort-arrow">⇅</span></th>
                    <th class="sortable" data-column="11">Rotated <span class="sort-arrow">⇅</span></th>
                    <th class="sortable" data-column="12">Type Capture <span class="sort-arrow">⇅</span></th>
                    <th class="sortable" data-column="13">Temps <span class="sort-arrow">⇅</span></th>
                </tr>
            </thead>
            <tbody>
                {''.join(rows)}
            </tbody>
        </table>
        """
    
    def _process_gallery_item(self, page: Dict[str, Any]) -> str:
        """Traite un élément de galerie (pour traitement parallèle)"""
        try:
            flags = page['qa_flags']

            orientation_conf = float(flags.get('orientation_confidence', 0.0))
            rotation_angle = float(flags.get('orientation_angle', 0))
            deskew_angle = float(flags.get('deskew_angle', 0.0))
            deskew_conf = float(flags.get('deskew_confidence', 0.0))

            # Récupérer les métadonnées de transformation pour analyser les raisons
            transform_data = page.get('transform_data', {})
            transforms = transform_data.get('transforms', [])
            
            # Analyser les transformations pour déterminer les statuts
            crop_transform = next((t for t in transforms if t.get('transform_type') == 'crop'), None)
            deskew_transform = next((t for t in transforms if t.get('transform_type') == 'deskew'), None)
            rotation_transform = next((t for t in transforms if t.get('transform_type') == 'rotation'), None)
            capture_classification = next((t for t in transforms if t.get('transform_type') == 'capture_classification'), None)
            
            # Fonction pour déterminer le type de badge
            def get_badge_type(flag_name, flag_value, metadata=None):
                """
                Retourne le type de badge:
                - 'applied' (vert): Action appliquée avec succès
                - 'skipped' (orange): Action non appliquée à cause des seuils/config
                - 'info' (bleu): Information/indication
                - 'inactive' (gris): Non applicable/non actif
                - 'warning' (rouge): Problème détecté
                """
                if flag_name == 'Rotation':
                    if flags.get('rotated', False):
                        return 'applied', f"Rotation appliquée : {rotation_angle:.0f}°"
                    else:
                        return 'inactive', "Aucune rotation nécessaire"
                
                elif flag_name == 'Deskew':
                    if deskew_transform:
                        deskew_status = deskew_transform.get('params', {}).get('status', '')
                        if deskew_status == 'applied':
                            return 'applied', f"Deskew appliqué : {deskew_angle:.2f}° (conf: {deskew_conf:.2f})"
                        elif deskew_status == 'skipped_low_confidence':
                            return 'skipped', f"Deskew ignoré : confiance trop faible ({deskew_conf:.2f})"
                        elif deskew_status == 'skipped_small_angle':
                            return 'skipped', f"Deskew ignoré : angle trop petit ({deskew_angle:.2f}°)"
                    return 'inactive', "Deskew non appliqué"
                
                elif flag_name == 'Crop':
                    if crop_transform:
                        crop_status = crop_transform.get('params', {}).get('status', '')
                        if crop_status == 'cropped':
                            return 'applied', "Crop intelligent appliqué"
                        elif crop_status == 'already_cropped':
                            return 'skipped', "Crop ignoré : document déjà cadré"
                        elif crop_status == 'skipped_scan':
                            return 'skipped', "Crop ignoré : document scanné détecté"
                    return 'inactive', "Crop non appliqué"
                
                elif flag_name == 'Dewarp':
                    if flags.get('dewarp_applied', False):
                        return 'applied', "Correction de perspective appliquée"
                    return 'inactive', "Pas de correction de perspective"
                
                elif flag_name == 'Capture Type':
                    capture_type = flags.get('capture_type', 'UNKNOWN')
                    white_pct = flags.get('capture_white_percentage', 0.0)
                    if capture_type == 'SCAN':
                        return 'info', f"SCAN détecté ({white_pct:.1%} blanc)"
                    elif capture_type == 'PHOTO':
                        return 'info', f"PHOTO détectée ({white_pct:.1%} blanc)"
                    return 'inactive', "Type inconnu"
                
                # Flags d'avertissement
                elif flag_name == 'Low Conf':
                    if flags.get('low_confidence_orientation', False):
                        return 'warning', f"Confiance faible : {orientation_conf:.2f}"
                    return 'inactive', f"Confiance OK : {orientation_conf:.2f}"
                
                elif flag_name == 'Overcrop':
                    if flags.get('overcrop_risk', False):
                        return 'warning', "Risque de sur-recadrage"
                    return 'inactive', "Recadrage OK"
                
                elif flag_name == 'No Quad':
                    if flags.get('no_quad_detected', False):
                        return 'warning', "Quadrilatère non détecté"
                    return 'inactive', "Quadrilatère détecté"
                
                elif flag_name == 'Low Contrast':
                    if flags.get('low_contrast_after_enhance', False):
                        return 'warning', "Contraste insuffisant"
                    return 'inactive', "Contraste OK"
                
                elif flag_name == 'Too Small':
                    if flags.get('too_small_final', False):
                        return 'warning', "Résolution insuffisante"
                    return 'inactive', "Résolution OK"
                
                return 'inactive', ""
            
            # Générer les badges avec le nouveau système de couleurs
            flags_badges = []
            badge_configs = [
                ('Rotation', None),
                ('Deskew', None),
                ('Crop', None),
                ('Dewarp', None),
                ('Capture Type', None),
                ('Low Conf', None),
                ('Overcrop', None),
                ('No Quad', None),
                ('Low Contrast', None),
                ('Too Small', None),
            ]
            
            for badge_name, metadata in badge_configs:
                badge_type, tooltip = get_badge_type(badge_name, None, metadata)
                flags_badges.append(
                    f'<span class="flag-badge badge-{badge_type}" title="{tooltip}">{badge_name}</span>'
                )
            
            # Générer les chemins d'images (base64 ou chemins relatifs)
            # Augmenter la taille pour une meilleure lisibilité (800px pour haute résolution)
            source_img = self._image_to_base64(page['input_path'], max_size=800) if page['input_path'] else None
            output_img = self._image_to_base64(page['output_path'], max_size=800)
            mask_img = self._generate_mask_image(page, max_size=800) if page['transform_data'] else None
            
            source_html = f'<img src="data:image/jpeg;base64,{source_img}" alt="Source">' if source_img else '<p style="color: #999;">Non disponible</p>'
            mask_html = f'<img src="data:image/jpeg;base64,{mask_img}" alt="Masque">' if mask_img else '<p style="color: #999;">Non disponible</p>'
            output_html = f'<img src="data:image/jpeg;base64,{output_img}" alt="Final">' if output_img else '<p style="color: #999;">Non disponible</p>'
            
            # Créer des data attributes pour le mode comparaison
            source_data = f'data-source="{source_img}"' if source_img else ''
            mask_data = f'data-mask="{mask_img}"' if mask_img else ''
            output_data = f'data-output="{output_img}"' if output_img else ''
            
            return f"""
            <div class="gallery-item" {source_data} {mask_data} {output_data}>
                <h4>{page['page_name']}</h4>
                <button class="compare-button" onclick="openCompareModal(this)">
                    🔍 Comparer Côte à Côte
                </button>
                <div class="gallery-images">
                    <div>
                        <strong>Source</strong>
                        {source_html}
                    </div>
                    <div>
                        <strong>Masque/Contour</strong>
                        {mask_html}
                    </div>
                    <div>
                        <strong>Final</strong>
                        {output_html}
                    </div>
                </div>
                <div class="flags-list">
                    {''.join(flags_badges)}
                </div>
            </div>
            """
        except Exception as e:
            return f'<div class="gallery-item"><p>Erreur: {str(e)}</p></div>'
    
    def _generate_gallery(self) -> str:
        """Génère la galerie HTML avec lazy loading (scroll infini)"""
        if not self.pages_data:
            return '<p>Aucune donnée disponible</p>'
        
        # Charger les 10 premières pages immédiatement
        initial_batch_size = 10
        pages_to_process_initial = self.pages_data[:initial_batch_size]
        
        logger.info(f"Generation de la galerie initiale pour {len(pages_to_process_initial)} pages...")
        logger.info(f"   Total de pages: {len(self.pages_data)}")
        
        # Traitement parallèle des éléments de galerie initiaux
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(self._process_gallery_item, page): idx 
                      for idx, page in enumerate(pages_to_process_initial)}
            
            results = [None] * len(pages_to_process_initial)
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    logger.error(f"Erreur lors du traitement de la page {idx}", exc_info=True)
                    results[idx] = f'<div class="gallery-item"><p>Erreur</p></div>'
        
        gallery_items_initial = [r for r in results if r is not None]
        
        # Pré-générer TOUS les items de galerie (pour le lazy loading)
        logger.info(f"Pre-generation de tous les items de galerie ({len(self.pages_data)} pages)...")
        all_gallery_items_html = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(self._process_gallery_item, page): idx 
                      for idx, page in enumerate(self.pages_data)}
            
            results_all = [None] * len(self.pages_data)
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    results_all[idx] = future.result()
                except Exception as e:
                    logger.error(f"Erreur lors du traitement de la page {idx}", exc_info=True)
                    results_all[idx] = f'<div class="gallery-item"><p>Erreur: page {idx}</p></div>'
        
        all_gallery_items_html = [r if r is not None else '' for r in results_all]
        
        # Créer le conteneur avec les items initiaux et un loader
        # Le loader doit être visible si on a plus de pages à charger
        loader_display = 'block' if len(self.pages_data) > initial_batch_size else 'none'
        end_display = 'none' if len(self.pages_data) > initial_batch_size else 'block'
        
        gallery_html = f'''
        <div class="gallery" id="galleryContainer">
            {"".join(gallery_items_initial)}
        </div>
        <div id="galleryLoader" style="text-align: center; padding: 20px; display: {loader_display};">
            <div style="display: inline-block; width: 40px; height: 40px; border: 4px solid #f3f3f3; border-top: 4px solid #3498db; border-radius: 50%; animation: spin 1s linear infinite;"></div>
            <p>Chargement des pages suivantes...</p>
        </div>
        <div id="galleryEnd" style="text-align: center; padding: 20px; display: {end_display}; color: #7f8c8d;">
            <p>Toutes les pages ont ete chargees ({len(self.pages_data)} pages)</p>
        </div>
        
        <script id="galleryItemsData" type="application/json">
        {json.dumps(all_gallery_items_html, ensure_ascii=False)}
        </script>
        '''
        
        return gallery_html
    
    def _image_to_base64(self, image_path: str, max_size: int = 300) -> Optional[str]:
        """Convertit une image en base64 pour l'embedding HTML (avec cache)"""
        if not image_path or not os.path.exists(image_path):
            return None
        
        # Vérifier le cache
        cache_key = f"{image_path}_{max_size}"
        if cache_key in self._image_cache:
            return self._image_cache[cache_key]
        
        try:
            import base64
            img = cv2.imread(image_path)
            if img is None:
                return None
            
            # Redimensionner si trop grande
            h, w = img.shape[:2]
            if max(h, w) > max_size:
                scale = max_size / max(h, w)
                new_w, new_h = int(w * scale), int(h * scale)
                img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
            
            # Encoder en JPEG avec qualité plus élevée pour les grandes images
            quality = 90 if max_size >= 600 else 85
            _, buffer = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, quality])
            img_base64 = base64.b64encode(buffer).decode('utf-8')
            
            # Mettre en cache
            self._image_cache[cache_key] = img_base64
            return img_base64
        except Exception:
            return None
    
    def _generate_mask_image(self, page_data: Dict[str, Any], max_size: int = 250) -> Optional[str]:
        """Génère une image de masque/contour à partir des transformations (optimisé)"""
        try:
            input_path = page_data.get('input_path')
            if not input_path or not os.path.exists(input_path):
                return None
            
            # Vérifier le cache avec la taille
            cache_key = f"mask_{input_path}_{max_size}"
            if cache_key in self._image_cache:
                return self._image_cache[cache_key]
            
            img = cv2.imread(input_path)
            if img is None:
                return None
            
            transform_data = page_data.get('transform_data')
            if not transform_data:
                return None
            
            # Redimensionner d'abord pour accélérer
            h, w = img.shape[:2]
            scale = 1.0
            if max(h, w) > max_size:
                scale = max_size / max(h, w)
                new_w, new_h = int(w * scale), int(h * scale)
                img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
            
            # Chercher la transformation de crop pour dessiner le contour
            for transform in transform_data.get('transforms', []):
                if transform.get('transform_type') == 'crop':
                    params = transform.get('params', {})
                    if 'source_points' in params:
                        points = np.array(params['source_points'], dtype=np.int32)
                        # Ajuster les points à l'échelle
                        if scale != 1.0:
                            points = (points * scale).astype(np.int32)
                        # Dessiner le contour
                        mask = img.copy()
                        cv2.polylines(mask, [points], True, (0, 255, 0), 2)
                        # Encoder en JPEG pour réduire la taille
                        import base64
                        _, buffer = cv2.imencode('.jpg', mask, [cv2.IMWRITE_JPEG_QUALITY, 85])
                        result = base64.b64encode(buffer).decode('utf-8')
                        # Mettre en cache
                        self._image_cache[cache_key] = result
                        return result
            
            return None
        except Exception:
            return None

