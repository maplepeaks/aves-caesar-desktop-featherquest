"""Explicit crop API preserves full source pixels and normal prediction behavior."""
from unittest.mock import Mock,patch
import numpy as np
import pytest
from PIL import Image
from bird_desktop.inference.pipeline import BirdOnnxPipeline
from bird_desktop.inference.preprocess import BirdBox,BirdImagePreprocessor


def test_explicit_boxes_use_full_image_and_existing_classification_path():
    pipeline=BirdOnnxPipeline.__new__(BirdOnnxPipeline)
    pipeline._active_providers=Mock(return_value=['CPUExecutionProvider'])
    observed=[]
    def classify(jobs,metadata):
        for job in jobs:
            assert job.image.size==(4000,3000)
            with patch.object(BirdImagePreprocessor,'_classifier_tensor',side_effect=lambda crop,size:observed.append((crop.size,crop.getpixel((15,15)),size)) or np.zeros((3,size,size))):
                BirdImagePreprocessor._classifier_tensor_from_box(job,224)
        return [[dict(species_id='a',confidence=.99)]]
    pipeline._classify_many=classify
    with Image.new('RGB',(4000,3000),'red') as image:
        image.putpixel((105,205),(0,255,0))
        result=pipeline.classify_boxes(image,[BirdBox([100,200,500,600],.9)],40,-111,'2026:08:29 12:00:00',crop_padding=10)
    assert observed==[((420,420),(0,255,0),224)]
    assert result.birds[0]['box']==[100,200,500,600]
    assert result.birds[0]['classification'][0]['confidence']==.99


def test_invalid_or_empty_boxes():
    pipeline=BirdOnnxPipeline.__new__(BirdOnnxPipeline)
    pipeline._active_providers=lambda:[]
    pipeline._classify_many=lambda jobs,metadata:[]
    with Image.new('RGB',(100,100)) as image:
        assert pipeline.classify_boxes(image,[]).birds==[]
        with pytest.raises(ValueError):pipeline.classify_boxes(image,[BirdBox([-1,0,10,10],1)])
        with pytest.raises(ValueError):pipeline.classify_boxes(image,[BirdBox([0,0,float('nan'),10],1)])
