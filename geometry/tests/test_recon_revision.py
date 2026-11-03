import json
from pathlib import Path
import pytest
from atas_geometry.reconstruct.recon_revision import revision_l0
from atas_geometry.service.durable import file_hash, atomic_json
def case(tmp_path):
    raw=tmp_path/'r/s/shots/q/l0';raw.mkdir(parents=True)
    (raw/'video.mp4').write_bytes(b'old');(raw/'cameras.npz').write_bytes(b'camera')
    digest=file_hash(raw/'video.mp4')
    atomic_json(raw/'observation.json',{'original_video_sha256':digest})
    return raw,dict(run_id='r',story_id='s',shot_id='q',video_sha256=digest)

def test_same_movie_reuses_legacy(tmp_path):
    raw,p=case(tmp_path);assert revision_l0(tmp_path,p)==raw

def test_new_movie_gets_new_path_without_touching_old(tmp_path):
    raw,p=case(tmp_path);old=(raw/'video.mp4').read_bytes();p['video_sha256']='b'*64
    dest=revision_l0(tmp_path,p)
    assert dest==raw.parent/'recon_versions'/('b'*64)/'l0'
    assert (raw/'video.mp4').read_bytes()==old and not dest.exists()
    assert revision_l0(tmp_path,p)==dest

def test_accepted_movie_cannot_be_replaced(tmp_path,monkeypatch):
    raw,p=case(tmp_path)
    monkeypatch.setattr('atas_geometry.reconstruct.recon_revision.committed_observation',lambda *a:raw)
    p['video_sha256']='b'*64
    with pytest.raises(ValueError,match='immutable'):revision_l0(tmp_path,p)

def test_hash_cannot_be_path(tmp_path):
    _,p=case(tmp_path);p['video_sha256']='../../bad'
    with pytest.raises(ValueError):revision_l0(tmp_path,p)
