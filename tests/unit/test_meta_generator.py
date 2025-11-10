"""
Unit tests for meta_generator.py
"""

import pytest
import tempfile
import os
import json
from pathlib import Path
from src.utils.meta_generator import generate_meta_json, _calculate_global_statistics
from src.utils.qa_flags import QAFlags
from src.utils.transform_handler import TransformSequence, Transform


class TestCalculateGlobalStatistics:
    """Tests for _calculate_global_statistics()"""
    
    def test_calculate_statistics_empty_list(self):
        """Test _calculate_global_statistics() with empty list"""
        result = _calculate_global_statistics([])
        assert result == {}
    
    def test_calculate_statistics_single_page(self):
        """Test _calculate_global_statistics() with single page"""
        pages = [{
            'flags': {
                'low_confidence_orientation': False,
                'overcrop_risk': False,
                'no_quad_detected': False,
                'dewarp_applied': False,
                'low_contrast_after_enhance': False,
                'too_small_final': False,
                'processing_time': 1.5
            }
        }]
        
        result = _calculate_global_statistics(pages)
        
        assert result['orientation_accuracy'] == 1.0
        assert result['overcrop_risk_count'] == 0
        assert result['overcrop_risk_rate'] == 0.0
        assert result['avg_processing_time'] == 1.5
        assert result['pages_with_flags'] == 0
    
    def test_calculate_statistics_multiple_pages(self):
        """Test _calculate_global_statistics() with multiple pages"""
        pages = [
            {
                'flags': {
                    'low_confidence_orientation': False,
                    'overcrop_risk': True,
                    'no_quad_detected': False,
                    'dewarp_applied': False,
                    'low_contrast_after_enhance': False,
                    'too_small_final': False,
                    'processing_time': 1.0
                }
            },
            {
                'flags': {
                    'low_confidence_orientation': True,
                    'overcrop_risk': False,
                    'no_quad_detected': True,
                    'dewarp_applied': False,
                    'low_contrast_after_enhance': True,
                    'too_small_final': False,
                    'processing_time': 2.0
                }
            },
            {
                'flags': {
                    'low_confidence_orientation': False,
                    'overcrop_risk': False,
                    'no_quad_detected': False,
                    'dewarp_applied': True,
                    'low_contrast_after_enhance': False,
                    'too_small_final': True,
                    'processing_time': 1.5
                }
            }
        ]
        
        result = _calculate_global_statistics(pages)
        
        assert result['orientation_accuracy'] == 2/3  # 2 out of 3 have good orientation
        assert result['overcrop_risk_count'] == 1
        assert result['overcrop_risk_rate'] == 1/3
        assert result['no_quad_detected_count'] == 1
        assert result['no_quad_detected_rate'] == 1/3
        assert result['dewarp_applied_count'] == 1
        assert result['low_contrast_count'] == 1
        assert result['too_small_count'] == 1
        assert result['avg_processing_time'] == (1.0 + 2.0 + 1.5) / 3
        assert result['pages_with_flags'] == 3  # All pages have at least one flag


class TestGenerateMetaJson:
    """Tests for generate_meta_json()"""
    
    def test_generate_meta_json_empty_directory(self):
        """Test generate_meta_json() with empty directory"""
        with tempfile.TemporaryDirectory() as tmpdir:
            meta_file = os.path.join(tmpdir, "meta.json")
            
            result = generate_meta_json(tmpdir, meta_file)
            
            assert result['total_pages'] == 0
            assert result['pages'] == []
            assert 'statistics' in result
            assert os.path.exists(meta_file)
    
    def test_generate_meta_json_with_qa_files(self):
        """Test generate_meta_json() with QA files"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test structure
            page1_dir = os.path.join(tmpdir, "page1")
            os.makedirs(page1_dir)
            
            # Create image file
            image_path = os.path.join(page1_dir, "page1.png")
            import numpy as np
            import cv2
            test_image = np.ones((100, 100, 3), dtype=np.uint8) * 128
            cv2.imwrite(image_path, test_image)
            
            # Create QA file
            qa_path = os.path.join(page1_dir, "page1.qa.json")
            qa_flags = QAFlags(
                low_confidence_orientation=False,
                orientation_confidence=0.9,
                processing_time=1.5
            )
            from src.utils.qa_flags import save_qa_flags
            save_qa_flags(image_path, qa_flags)
            
            # Create transform file
            transform_sequence = TransformSequence(
                input_path="/input/page1.png",
                output_path=image_path
            )
            transform_sequence.add_transform(Transform(
                transform_type='rotation',
                params={'angle': 90},
                order=0
            ))
            from src.utils.transform_handler import save_transforms
            save_transforms(image_path, transform_sequence)
            
            # Generate meta.json
            meta_file = os.path.join(tmpdir, "meta.json")
            result = generate_meta_json(tmpdir, meta_file)
            
            assert result['total_pages'] == 1
            assert len(result['pages']) == 1
            assert result['pages'][0]['page_name'] == 'page1'
            assert 'flags' in result['pages'][0]
            assert 'transforms' in result['pages'][0]
            assert 'statistics' in result
            assert os.path.exists(meta_file)
            
            # Verify JSON file content
            with open(meta_file, 'r') as f:
                meta_data = json.load(f)
            assert meta_data['total_pages'] == 1
    
    def test_generate_meta_json_missing_image(self):
        """Test generate_meta_json() when QA file exists but image doesn't"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create QA file without corresponding image
            qa_path = os.path.join(tmpdir, "missing.qa.json")
            qa_flags = QAFlags()
            import json
            with open(qa_path, 'w') as f:
                json.dump(qa_flags.to_dict(), f)
            
            meta_file = os.path.join(tmpdir, "meta.json")
            result = generate_meta_json(tmpdir, meta_file)
            
            # Should skip pages without images
            assert result['total_pages'] == 1  # QA file found
            assert len(result['pages']) == 0  # But no valid page entry
    
    def test_generate_meta_json_multiple_pages(self):
        """Test generate_meta_json() with multiple pages"""
        with tempfile.TemporaryDirectory() as tmpdir:
            import numpy as np
            import cv2
            
            # Create multiple pages
            for i in range(3):
                page_dir = os.path.join(tmpdir, f"page{i+1}")
                os.makedirs(page_dir)
                
                # Create image
                image_path = os.path.join(page_dir, f"page{i+1}.png")
                test_image = np.ones((100, 100, 3), dtype=np.uint8) * (100 + i * 50)
                cv2.imwrite(image_path, test_image)
                
                # Create QA file
                qa_flags = QAFlags(
                    processing_time=1.0 + i * 0.5,
                    low_confidence_orientation=(i == 1)  # Page 2 has low confidence
                )
                from src.utils.qa_flags import save_qa_flags
                save_qa_flags(image_path, qa_flags)
            
            meta_file = os.path.join(tmpdir, "meta.json")
            result = generate_meta_json(tmpdir, meta_file)
            
            assert result['total_pages'] == 3
            assert len(result['pages']) == 3
            assert 'statistics' in result
            assert result['statistics']['orientation_accuracy'] == 2/3  # 2 out of 3 have good orientation

