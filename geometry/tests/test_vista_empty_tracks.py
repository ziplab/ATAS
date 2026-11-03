import importlib.util
import json
from pathlib import Path
import numpy as np
import pytest
import os

source=Path(__file__).resolve().parents[1]/'backbones/scripts/preprocess/avwm_entity_mask_export.py'
spec=importlib.util.spec_from_file_location('candidate_export',source)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)

def test_empty_tracks_are_explicit(tmp_path):
    module.export_entity_masks([[],[],[]],tmp_path,frame_shape=(24,32))
    saved=json.loads((tmp_path/'instance_tracks.json').read_text())
    assert saved['frame_count']==3
    assert (saved['height'],saved['width'])==(24,32)
    assert saved['tracks']=={}
    assert saved['segmentation_status']=='no_tracks_detected'

def test_low_confidence_is_not_promoted(tmp_path):
    module.export_entity_masks([[dict(id=1,score=.1,keyword='cat',mask=np.ones((24,32),bool))]],
                               tmp_path,frame_shape=(24,32))
    assert json.loads((tmp_path/'instance_tracks.json').read_text())['tracks']=={}

def test_detected_track_preserves_all_frames(tmp_path):
    module.export_entity_masks([[dict(id=1,score=.9,keyword='cat',mask=np.ones((24,32),bool))],[]],
                               tmp_path,frame_shape=(24,32))
    saved=json.loads((tmp_path/'instance_tracks.json').read_text())
    track=next(iter(saved['tracks'].values()))
    assert track['frame_count']==2
    assert not track['identity_verified']
    assert len(list((tmp_path/track['mask_dir']).glob('*.png')))==2

def test_missing_dimensions_is_not_invented(tmp_path):
    with pytest.raises(ValueError,match='frame_shape'):
        module.export_entity_masks([[]],tmp_path)
