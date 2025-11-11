"""
Tests unitaires pour GeometryNormalizer, notamment la parallélisation avec ProcessPoolExecutor
"""

import pytest
import numpy as np
import cv2
import tempfile
import os
from pathlib import Path
from unittest.mock import patch, Mock, MagicMock, call
from dataclasses import asdict

from src.pipeline.geometry import GeometryNormalizer, init_geometry_worker, process_single_image_geometry
from src.pipeline.models import GeometryOutput, CropMetadata, DeskewMetadata
from src.models.registry import ModelRegistry


class TestGeometryNormalizerMultiprocessing:
    """Tests pour la parallélisation avec ProcessPoolExecutor"""
    
    @patch('src.pipeline.geometry.ProcessPoolExecutor')
    @patch('src.pipeline.geometry.get_files')
    @patch('src.pipeline.geometry.get_output_path')
    def test_process_batch_uses_process_pool_executor(
        self,
        mock_get_output_path,
        mock_get_files,
        mock_process_pool_executor,
        mock_geometry_config,
        mock_qa_config,
        mock_performance_config,
        mock_output_config,
        mock_model_registry
    ):
        """
        Test que process_batch utilise bien ProcessPoolExecutor pour la parallélisation.
        
        On ne teste pas le multiprocessing lui-même, mais que le code l'appelle correctement.
        """
        # Configuration des mocks
        mock_get_files.return_value = [
            '/tmp/test1.png',
            '/tmp/test2.png',
            '/tmp/test3.png'
        ]
        
        def mock_output_path_side_effect(file_path, output_dir):
            filename = Path(file_path).name
            return str(Path(output_dir) / filename)
        
        mock_get_output_path.side_effect = mock_output_path_side_effect
        
        # Mock du ProcessPoolExecutor
        mock_executor_instance = MagicMock()
        mock_executor_context = MagicMock()
        mock_executor_context.__enter__.return_value = mock_executor_instance
        mock_executor_context.__exit__.return_value = None
        mock_process_pool_executor.return_value = mock_executor_context
        
        # Mock des résultats du ProcessPoolExecutor
        mock_result1 = GeometryOutput(
            status='success',
            input_path='/tmp/test1.png',
            output_path='',
            output_transformed_path='/tmp/output/test1_transformed.png',
            transform_file='',
            qa_file='',
            crop_applied=True,
            crop_metadata=CropMetadata(
                crop_applied=True,
                area_ratio=0.8,
                status='cropped'
            ),
            deskew_applied=False,
            deskew_angle=0.0,
            deskew_metadata=DeskewMetadata(
                deskew_applied=False,
                angle=0.0,
                confidence=0.0,
                status='skipped'
            ),
            orientation_detected=True,
            angle=0,
            rotation_applied=False,
            transforms={},
            qa_flags={},
            processing_time=1.0
        )
        mock_result2 = GeometryOutput(
            status='success',
            input_path='/tmp/test2.png',
            output_path='',
            output_transformed_path='/tmp/output/test2_transformed.png',
            transform_file='',
            qa_file='',
            crop_applied=True,
            crop_metadata=CropMetadata(
                crop_applied=True,
                area_ratio=0.75,
                status='cropped'
            ),
            deskew_applied=False,
            deskew_angle=0.0,
            deskew_metadata=DeskewMetadata(
                deskew_applied=False,
                angle=0.0,
                confidence=0.0,
                status='skipped'
            ),
            orientation_detected=True,
            angle=0,
            rotation_applied=False,
            transforms={},
            qa_flags={},
            processing_time=1.0
        )
        mock_result3 = GeometryOutput(
            status='success',
            input_path='/tmp/test3.png',
            output_path='',
            output_transformed_path='/tmp/output/test3_transformed.png',
            transform_file='',
            qa_file='',
            crop_applied=True,
            crop_metadata=CropMetadata(
                crop_applied=True,
                area_ratio=0.9,
                status='cropped'
            ),
            deskew_applied=False,
            deskew_angle=0.0,
            deskew_metadata=DeskewMetadata(
                deskew_applied=False,
                angle=0.0,
                confidence=0.0,
                status='skipped'
            ),
            orientation_detected=True,
            angle=0,
            rotation_applied=False,
            transforms={},
            qa_flags={},
            processing_time=1.0
        )
        
        mock_executor_instance.map.return_value = [mock_result1, mock_result2, mock_result3]
        
        # Créer le normalizer
        normalizer = GeometryNormalizer(
            geo_config=mock_geometry_config,
            qa_config=mock_qa_config,
            perf_config=mock_performance_config,
            output_config=mock_output_config,
            model_registry=mock_model_registry
        )
        
        # Mock intelligent_crop_batch pour retourner des images et métadonnées
        def mock_intelligent_crop_batch(images, capture_types):
            # Retourner les images telles quelles et des métadonnées de crop mockées
            crop_metadata_list = []
            for i in range(len(images)):
                crop_metadata_list.append({
                    'crop_applied': True,
                    'area_ratio': 0.8,
                    'status': 'cropped'
                })
            return images, crop_metadata_list
        
        normalizer.intelligent_crop_batch = mock_intelligent_crop_batch
        
        # Créer un répertoire temporaire pour les tests
        with tempfile.TemporaryDirectory() as tmpdir:
            input_dir = Path(tmpdir) / "input"
            output_dir = Path(tmpdir) / "output"
            input_dir.mkdir()
            output_dir.mkdir()
            
            # Créer des fichiers images de test
            test_images = []
            for i, file_path in enumerate(mock_get_files.return_value):
                img = np.ones((100, 100, 3), dtype=np.uint8) * 255
                test_file_path = input_dir / Path(file_path).name
                cv2.imwrite(str(test_file_path), img)
                test_images.append(img)
            
            # Mocker cv2.imread pour retourner nos images de test
            original_imread = cv2.imread
            def mock_imread(path):
                filename = Path(path).name
                for i, file_path in enumerate(mock_get_files.return_value):
                    if Path(file_path).name == filename:
                        return test_images[i]
                return original_imread(path)
            
            with patch('cv2.imread', side_effect=mock_imread):
                # Exécuter process_batch
                results = normalizer.process_batch(str(input_dir), str(output_dir))
            
            # Vérifications
            # 1. ProcessPoolExecutor a été instancié
            mock_process_pool_executor.assert_called_once()
            
            # 2. Vérifier les arguments de ProcessPoolExecutor
            call_args = mock_process_pool_executor.call_args
            assert call_args.kwargs['max_workers'] == mock_performance_config.max_workers
            assert call_args.kwargs['initializer'] == init_geometry_worker
            assert 'initargs' in call_args.kwargs
            
            # 3. Vérifier que initargs contient les configurations sérialisées
            initargs = call_args.kwargs['initargs']
            assert len(initargs) == 4
            assert isinstance(initargs[0], dict)  # geo_config_dict
            assert isinstance(initargs[1], dict)  # qa_config_dict
            assert isinstance(initargs[2], dict)  # perf_config_dict
            assert isinstance(initargs[3], dict)  # output_config_dict
            
            # Vérifier que les configurations contiennent les bonnes clés
            geo_dict = initargs[0]
            assert 'crop_enabled' in geo_dict
            assert 'deskew_enabled' in geo_dict
            assert geo_dict['crop_enabled'] == mock_geometry_config.crop_enabled
            
            perf_dict = initargs[2]
            assert 'max_workers' in perf_dict
            assert perf_dict['max_workers'] == mock_performance_config.max_workers
            assert 'parallelization_threshold' in perf_dict
            assert perf_dict['parallelization_threshold'] == mock_performance_config.parallelization_threshold
            
            # 4. Vérifier que map a été appelé avec process_single_image_geometry
            mock_executor_instance.map.assert_called_once()
            map_call_args = mock_executor_instance.map.call_args
            assert map_call_args[0][0] == process_single_image_geometry
            tasks_passed = map_call_args[0][1]
            assert len(tasks_passed) == 3  # 3 tâches
            
            # 5. Vérifier la structure des tâches passées au worker
            for task in tasks_passed:
                assert len(task) == 5  # (image, metadata_info, output_path, original_path, crop_metadata)
                assert isinstance(task[0], np.ndarray)  # image
                assert isinstance(task[1], dict)  # metadata_info
                assert isinstance(task[2], str)  # output_path
                assert isinstance(task[3], str)  # original_path
                assert isinstance(task[4], dict)  # crop_metadata
            
            # 6. Vérifier que les résultats sont retournés
            assert len(results) == 3
            assert all(isinstance(r, GeometryOutput) for r in results)
    
    @patch('src.pipeline.geometry.ProcessPoolExecutor')
    @patch('src.pipeline.geometry.get_files')
    @patch('src.pipeline.geometry.get_output_path')
    def test_process_batch_sequential_for_small_batches(
        self,
        mock_get_output_path,
        mock_get_files,
        mock_process_pool_executor,
        mock_geometry_config,
        mock_qa_config,
        mock_performance_config,
        mock_output_config,
        mock_model_registry
    ):
        """
        Test que process_batch utilise le traitement séquentiel pour les petits lots (< seuil de 2).
        
        Pour un PDF d'une seule page, l'overhead de création de ProcessPoolExecutor
        dépasse le gain de parallélisation, donc on doit traiter séquentiellement.
        """
        # Configuration des mocks - 1 seul fichier (en dessous du seuil de 2)
        mock_get_files.return_value = [
            '/tmp/test1.png'
        ]
        
        def mock_output_path_side_effect(file_path, output_dir):
            filename = Path(file_path).name
            return str(Path(output_dir) / filename)
        
        mock_get_output_path.side_effect = mock_output_path_side_effect
        
        # Créer le normalizer
        normalizer = GeometryNormalizer(
            geo_config=mock_geometry_config,
            qa_config=mock_qa_config,
            perf_config=mock_performance_config,
            output_config=mock_output_config,
            model_registry=mock_model_registry
        )
        
        # Mock intelligent_crop_batch pour retourner des images et métadonnées
        def mock_intelligent_crop_batch(images, capture_types):
            crop_metadata_list = []
            for i in range(len(images)):
                crop_metadata_list.append({
                    'crop_applied': True,
                    'area_ratio': 0.8,
                    'status': 'cropped'
                })
            return images, crop_metadata_list
        
        normalizer.intelligent_crop_batch = mock_intelligent_crop_batch
        
        # Mock de process pour vérifier qu'il est appelé séquentiellement
        mock_result = GeometryOutput(
            status='success',
            input_path='/tmp/test1.png',
            output_path='',
            output_transformed_path='/tmp/output/test1_transformed.png',
            transform_file='',
            qa_file='',
            crop_applied=True,
            crop_metadata=CropMetadata(
                crop_applied=True,
                area_ratio=0.8,
                status='cropped'
            ),
            deskew_applied=False,
            deskew_angle=0.0,
            deskew_metadata=DeskewMetadata(
                deskew_applied=False,
                angle=0.0,
                confidence=0.0,
                status='skipped'
            ),
            orientation_detected=True,
            angle=0,
            rotation_applied=False,
            transforms={},
            qa_flags={},
            processing_time=1.0
        )
        
        normalizer.process = Mock(return_value=mock_result)
        
        # Créer un répertoire temporaire pour les tests
        with tempfile.TemporaryDirectory() as tmpdir:
            input_dir = Path(tmpdir) / "input"
            output_dir = Path(tmpdir) / "output"
            input_dir.mkdir()
            output_dir.mkdir()
            
            # Créer un fichier image de test
            img = np.ones((100, 100, 3), dtype=np.uint8) * 255
            test_file_path = input_dir / "test1.png"
            cv2.imwrite(str(test_file_path), img)
            
            # Mocker cv2.imread
            with patch('cv2.imread', return_value=img):
                # Exécuter process_batch
                results = normalizer.process_batch(str(input_dir), str(output_dir))
            
            # Vérifications
            # 1. ProcessPoolExecutor NE DOIT PAS être appelé pour 1 fichier
            mock_process_pool_executor.assert_not_called()
            
            # 2. Le traitement séquentiel a été utilisé (normalizer.process doit avoir été appelé)
            assert normalizer.process.call_count == 1
            
            # 3. Les résultats sont retournés
            assert len(results) == 1
            assert all(isinstance(r, GeometryOutput) for r in results)
            assert results[0].status == 'success'
    
    @patch('src.pipeline.geometry.ProcessPoolExecutor')
    @patch('src.pipeline.geometry.get_files')
    @patch('src.pipeline.geometry.get_output_path')
    def test_process_batch_parallel_for_large_batches(
        self,
        mock_get_output_path,
        mock_get_files,
        mock_process_pool_executor,
        mock_geometry_config,
        mock_qa_config,
        mock_performance_config,
        mock_output_config,
        mock_model_registry
    ):
        """
        Test que process_batch utilise le traitement parallèle pour les gros lots (>= seuil de 2).
        
        Pour 2 fichiers ou plus, le traitement parallèle est utilisé avec ProcessPoolExecutor.
        """
        # Configuration des mocks - 2 fichiers (au-dessus du seuil de 2)
        mock_get_files.return_value = [
            '/tmp/test1.png',
            '/tmp/test2.png'
        ]
        
        def mock_output_path_side_effect(file_path, output_dir):
            filename = Path(file_path).name
            return str(Path(output_dir) / filename)
        
        mock_get_output_path.side_effect = mock_output_path_side_effect
        
        # Mock du ProcessPoolExecutor
        mock_executor_instance = MagicMock()
        mock_executor_context = MagicMock()
        mock_executor_context.__enter__.return_value = mock_executor_instance
        mock_executor_context.__exit__.return_value = None
        mock_process_pool_executor.return_value = mock_executor_context
        
        # Mock des résultats du ProcessPoolExecutor
        mock_result1 = GeometryOutput(
            status='success',
            input_path='/tmp/test1.png',
            output_path='',
            output_transformed_path='/tmp/output/test1_transformed.png',
            transform_file='',
            qa_file='',
            crop_applied=True,
            crop_metadata=CropMetadata(
                crop_applied=True,
                area_ratio=0.8,
                status='cropped'
            ),
            deskew_applied=False,
            deskew_angle=0.0,
            deskew_metadata=DeskewMetadata(
                deskew_applied=False,
                angle=0.0,
                confidence=0.0,
                status='skipped'
            ),
            orientation_detected=True,
            angle=0,
            rotation_applied=False,
            transforms={},
            qa_flags={},
            processing_time=1.0
        )
        mock_result2 = GeometryOutput(
            status='success',
            input_path='/tmp/test2.png',
            output_path='',
            output_transformed_path='/tmp/output/test2_transformed.png',
            transform_file='',
            qa_file='',
            crop_applied=True,
            crop_metadata=CropMetadata(
                crop_applied=True,
                area_ratio=0.75,
                status='cropped'
            ),
            deskew_applied=False,
            deskew_angle=0.0,
            deskew_metadata=DeskewMetadata(
                deskew_applied=False,
                angle=0.0,
                confidence=0.0,
                status='skipped'
            ),
            orientation_detected=True,
            angle=0,
            rotation_applied=False,
            transforms={},
            qa_flags={},
            processing_time=1.0
        )
        
        mock_executor_instance.map.return_value = [mock_result1, mock_result2]
        
        # Créer le normalizer
        normalizer = GeometryNormalizer(
            geo_config=mock_geometry_config,
            qa_config=mock_qa_config,
            perf_config=mock_performance_config,
            output_config=mock_output_config,
            model_registry=mock_model_registry
        )
        
        # Mock intelligent_crop_batch pour retourner des images et métadonnées
        def mock_intelligent_crop_batch(images, capture_types):
            crop_metadata_list = []
            for i in range(len(images)):
                crop_metadata_list.append({
                    'crop_applied': True,
                    'area_ratio': 0.8,
                    'status': 'cropped'
                })
            return images, crop_metadata_list
        
        normalizer.intelligent_crop_batch = mock_intelligent_crop_batch
        
        # Créer un répertoire temporaire pour les tests
        with tempfile.TemporaryDirectory() as tmpdir:
            input_dir = Path(tmpdir) / "input"
            output_dir = Path(tmpdir) / "output"
            input_dir.mkdir()
            output_dir.mkdir()
            
            # Créer des fichiers images de test
            test_images = []
            for i, file_path in enumerate(mock_get_files.return_value):
                img = np.ones((100, 100, 3), dtype=np.uint8) * 255
                test_file_path = input_dir / Path(file_path).name
                cv2.imwrite(str(test_file_path), img)
                test_images.append(img)
            
            # Mocker cv2.imread
            original_imread = cv2.imread
            def mock_imread(path):
                filename = Path(path).name
                for i, file_path in enumerate(mock_get_files.return_value):
                    if Path(file_path).name == filename:
                        return test_images[i]
                return original_imread(path)
            
            with patch('cv2.imread', side_effect=mock_imread):
                # Exécuter process_batch
                results = normalizer.process_batch(str(input_dir), str(output_dir))
            
            # Vérifications
            # 1. ProcessPoolExecutor DOIT être appelé pour 2 fichiers ou plus
            mock_process_pool_executor.assert_called_once()
            
            # 2. Vérifier que map a été appelé avec process_single_image_geometry
            mock_executor_instance.map.assert_called_once()
            map_call_args = mock_executor_instance.map.call_args
            assert map_call_args[0][0] == process_single_image_geometry
            tasks_passed = map_call_args[0][1]
            assert len(tasks_passed) == 2  # 2 tâches
            
            # 3. Vérifier que les résultats sont retournés
            assert len(results) == 2
            assert all(isinstance(r, GeometryOutput) for r in results)
    
    @patch('src.pipeline.geometry.ProcessPoolExecutor')
    @patch('src.pipeline.geometry.get_files')
    @patch('src.pipeline.geometry.get_output_path')
    def test_process_batch_error_handling_partial_failure(
        self,
        mock_get_output_path,
        mock_get_files,
        mock_process_pool_executor,
        mock_geometry_config,
        mock_qa_config,
        mock_performance_config,
        mock_output_config,
        mock_model_registry
    ):
        """
        Test que si une page échoue, les autres pages continuent d'être traitées.
        
        Scénario : Un PDF de 3 pages, la page 2 échoue, mais les pages 1 et 3 réussissent.
        """
        # Configuration des mocks
        mock_get_files.return_value = [
            '/tmp/test1.png',
            '/tmp/test2.png',
            '/tmp/test3.png'
        ]
        
        def mock_output_path_side_effect(file_path, output_dir):
            filename = Path(file_path).name
            return str(Path(output_dir) / filename)
        
        mock_get_output_path.side_effect = mock_output_path_side_effect
        
        # Mock du ProcessPoolExecutor
        mock_executor_instance = MagicMock()
        mock_executor_context = MagicMock()
        mock_executor_context.__enter__.return_value = mock_executor_instance
        mock_executor_context.__exit__.return_value = None
        mock_process_pool_executor.return_value = mock_executor_context
        
        # Mock des résultats : page 1 réussit, page 2 échoue, page 3 réussit
        mock_result1 = GeometryOutput(
            status='success',
            input_path='/tmp/test1.png',
            output_path='',
            output_transformed_path='/tmp/output/test1_transformed.png',
            transform_file='',
            qa_file='',
            crop_applied=True,
            crop_metadata=CropMetadata(
                crop_applied=True,
                area_ratio=0.8,
                status='cropped'
            ),
            deskew_applied=False,
            deskew_angle=0.0,
            deskew_metadata=DeskewMetadata(
                deskew_applied=False,
                angle=0.0,
                confidence=0.0,
                status='skipped'
            ),
            orientation_detected=True,
            angle=0,
            rotation_applied=False,
            transforms={},
            qa_flags={},
            processing_time=1.0
        )
        
        # Page 2 échoue
        mock_result2 = GeometryOutput(
            status='error',
            input_path='/tmp/test2.png',
            output_path='',
            output_transformed_path='',
            transform_file='',
            qa_file='',
            crop_applied=False,
            crop_metadata=CropMetadata(
                crop_applied=False,
                area_ratio=0.0,
                status='error'
            ),
            deskew_applied=False,
            deskew_angle=0.0,
            deskew_metadata=DeskewMetadata(
                deskew_applied=False,
                angle=0.0,
                confidence=0.0,
                status='error'
            ),
            orientation_detected=False,
            angle=0,
            rotation_applied=False,
            transforms={},
            qa_flags={},
            processing_time=0.0,
            error='Erreur de deskew simulée'
        )
        
        mock_result3 = GeometryOutput(
            status='success',
            input_path='/tmp/test3.png',
            output_path='',
            output_transformed_path='/tmp/output/test3_transformed.png',
            transform_file='',
            qa_file='',
            crop_applied=True,
            crop_metadata=CropMetadata(
                crop_applied=True,
                area_ratio=0.9,
                status='cropped'
            ),
            deskew_applied=False,
            deskew_angle=0.0,
            deskew_metadata=DeskewMetadata(
                deskew_applied=False,
                angle=0.0,
                confidence=0.0,
                status='skipped'
            ),
            orientation_detected=True,
            angle=0,
            rotation_applied=False,
            transforms={},
            qa_flags={},
            processing_time=1.0
        )
        
        mock_executor_instance.map.return_value = [mock_result1, mock_result2, mock_result3]
        
        # Créer le normalizer
        normalizer = GeometryNormalizer(
            geo_config=mock_geometry_config,
            qa_config=mock_qa_config,
            perf_config=mock_performance_config,
            output_config=mock_output_config,
            model_registry=mock_model_registry
        )
        
        # Mock intelligent_crop_batch
        def mock_intelligent_crop_batch(images, capture_types):
            crop_metadata_list = []
            for i in range(len(images)):
                crop_metadata_list.append({
                    'crop_applied': True,
                    'area_ratio': 0.8,
                    'status': 'cropped'
                })
            return images, crop_metadata_list
        
        normalizer.intelligent_crop_batch = mock_intelligent_crop_batch
        
        # Créer un répertoire temporaire pour les tests
        with tempfile.TemporaryDirectory() as tmpdir:
            input_dir = Path(tmpdir) / "input"
            output_dir = Path(tmpdir) / "output"
            input_dir.mkdir()
            output_dir.mkdir()
            
            # Créer des fichiers images de test
            test_images = []
            for i, file_path in enumerate(mock_get_files.return_value):
                img = np.ones((100, 100, 3), dtype=np.uint8) * 255
                test_file_path = input_dir / Path(file_path).name
                cv2.imwrite(str(test_file_path), img)
                test_images.append(img)
            
            # Mocker cv2.imread
            original_imread = cv2.imread
            def mock_imread(path):
                filename = Path(path).name
                for i, file_path in enumerate(mock_get_files.return_value):
                    if Path(file_path).name == filename:
                        return test_images[i]
                return original_imread(path)
            
            with patch('cv2.imread', side_effect=mock_imread):
                # Exécuter process_batch
                results = normalizer.process_batch(str(input_dir), str(output_dir))
            
            # Vérifications
            # 1. Tous les résultats sont retournés (même ceux en erreur)
            assert len(results) == 3
            
            # 2. Page 1 réussit
            assert results[0].status == 'success'
            assert results[0].input_path == '/tmp/test1.png'
            
            # 3. Page 2 échoue
            assert results[1].status == 'error'
            assert results[1].input_path == '/tmp/test2.png'
            assert 'error' in results[1].error.lower() or results[1].error == 'Erreur de deskew simulée'
            
            # 4. Page 3 réussit
            assert results[2].status == 'success'
            assert results[2].input_path == '/tmp/test3.png'
            
            # 5. Le ProcessPoolExecutor a bien été utilisé
            mock_process_pool_executor.assert_called_once()
            mock_executor_instance.map.assert_called_once()
    
    @patch('src.pipeline.geometry.ProcessPoolExecutor')
    @patch('src.pipeline.geometry.get_files')
    @patch('src.pipeline.geometry.get_output_path')
    def test_process_batch_fallback_to_sequential_on_executor_error(
        self,
        mock_get_output_path,
        mock_get_files,
        mock_process_pool_executor,
        mock_geometry_config,
        mock_qa_config,
        mock_performance_config,
        mock_output_config,
        mock_model_registry
    ):
        """
        Test que si ProcessPoolExecutor échoue complètement, on bascule vers le traitement séquentiel.
        
        Scénario : ProcessPoolExecutor lève une exception, le code doit basculer vers le traitement séquentiel.
        """
        # Configuration des mocks
        mock_get_files.return_value = [
            '/tmp/test1.png',
            '/tmp/test2.png'
        ]
        
        def mock_output_path_side_effect(file_path, output_dir):
            filename = Path(file_path).name
            return str(Path(output_dir) / filename)
        
        mock_get_output_path.side_effect = mock_output_path_side_effect
        
        # Mock du ProcessPoolExecutor qui lève une exception
        mock_executor_context = MagicMock()
        mock_executor_context.__enter__.side_effect = RuntimeError("Erreur d'initialisation du ProcessPoolExecutor")
        mock_process_pool_executor.return_value = mock_executor_context
        
        # Créer le normalizer
        normalizer = GeometryNormalizer(
            geo_config=mock_geometry_config,
            qa_config=mock_qa_config,
            perf_config=mock_performance_config,
            output_config=mock_output_config,
            model_registry=mock_model_registry
        )
        
        # Mock intelligent_crop_batch
        def mock_intelligent_crop_batch(images, capture_types):
            crop_metadata_list = []
            for i in range(len(images)):
                crop_metadata_list.append({
                    'crop_applied': True,
                    'area_ratio': 0.8,
                    'status': 'cropped'
                })
            return images, crop_metadata_list
        
        normalizer.intelligent_crop_batch = mock_intelligent_crop_batch
        
        # Mock process pour simuler le traitement séquentiel
        mock_result = GeometryOutput(
            status='success',
            input_path='/tmp/test1.png',
            output_path='',
            output_transformed_path='/tmp/output/test1_transformed.png',
            transform_file='',
            qa_file='',
            crop_applied=True,
            crop_metadata=CropMetadata(
                crop_applied=True,
                area_ratio=0.8,
                status='cropped'
            ),
            deskew_applied=False,
            deskew_angle=0.0,
            deskew_metadata=DeskewMetadata(
                deskew_applied=False,
                angle=0.0,
                confidence=0.0,
                status='skipped'
            ),
            orientation_detected=True,
            angle=0,
            rotation_applied=False,
            transforms={},
            qa_flags={},
            processing_time=1.0
        )
        
        normalizer.process = Mock(return_value=mock_result)
        
        # Créer un répertoire temporaire pour les tests
        with tempfile.TemporaryDirectory() as tmpdir:
            input_dir = Path(tmpdir) / "input"
            output_dir = Path(tmpdir) / "output"
            input_dir.mkdir()
            output_dir.mkdir()
            
            # Créer des fichiers images de test
            test_images = []
            for i, file_path in enumerate(mock_get_files.return_value):
                img = np.ones((100, 100, 3), dtype=np.uint8) * 255
                test_file_path = input_dir / Path(file_path).name
                cv2.imwrite(str(test_file_path), img)
                test_images.append(img)
            
            # Mocker cv2.imread
            original_imread = cv2.imread
            def mock_imread(path):
                filename = Path(path).name
                for i, file_path in enumerate(mock_get_files.return_value):
                    if Path(file_path).name == filename:
                        return test_images[i]
                return original_imread(path)
            
            with patch('cv2.imread', side_effect=mock_imread):
                # Exécuter process_batch
                results = normalizer.process_batch(str(input_dir), str(output_dir))
            
            # Vérifications
            # 1. ProcessPoolExecutor a été appelé
            mock_process_pool_executor.assert_called_once()
            
            # 2. Le traitement séquentiel a été utilisé en fallback
            # (normalizer.process devrait avoir été appelé pour chaque image)
            assert normalizer.process.call_count == 2
            
            # 3. Les résultats sont retournés
            assert len(results) == 2
            assert all(isinstance(r, GeometryOutput) for r in results)


class TestGeometryWorkerFunctions:
    """Tests pour les fonctions worker du multiprocessing"""
    
    def test_init_geometry_worker_creates_normalizer(self, mock_geometry_config, mock_qa_config, mock_performance_config, mock_output_config):
        """
        Test que init_geometry_worker crée correctement un GeometryNormalizer.
        """
        # Convertir les configurations en dictionnaires
        geo_config_dict = asdict(mock_geometry_config)
        qa_config_dict = asdict(mock_qa_config)
        perf_config_dict = asdict(mock_performance_config)
        output_config_dict = asdict(mock_output_config)
        
        # Appeler init_geometry_worker
        init_geometry_worker(geo_config_dict, qa_config_dict, perf_config_dict, output_config_dict)
        
        # Vérifier que worker_normalizer a été créé
        from src.pipeline.geometry import worker_normalizer
        assert worker_normalizer is not None
        assert isinstance(worker_normalizer, GeometryNormalizer)
    
    def test_process_single_image_geometry_without_init_raises_error(self):
        """
        Test que process_single_image_geometry lève une erreur si le worker n'est pas initialisé.
        """
        # Réinitialiser worker_normalizer
        import src.pipeline.geometry as geometry_module
        original_worker = geometry_module.worker_normalizer
        geometry_module.worker_normalizer = None
        
        try:
            # Créer une tâche de test
            image = np.ones((100, 100, 3), dtype=np.uint8) * 255
            metadata_info = {
                'capture_type': 'PHOTO',
                'capture_info': None,
                'original_input_path': '/tmp/test.png'
            }
            output_path = '/tmp/output/test_transformed.png'
            original_path = '/tmp/test.png'
            crop_metadata = {
                'crop_applied': True,
                'area_ratio': 0.8,
                'status': 'cropped'
            }
            
            task = (image, metadata_info, output_path, original_path, crop_metadata)
            
            # Vérifier que ça lève une RuntimeError
            with pytest.raises(RuntimeError, match="Worker normalizer non initialisé"):
                process_single_image_geometry(task)
        finally:
            # Restaurer le worker original
            geometry_module.worker_normalizer = original_worker

