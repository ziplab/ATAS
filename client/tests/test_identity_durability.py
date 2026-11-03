import json
from pathlib import Path
from types import SimpleNamespace
import pytest
from atas.world.identity import appearance_pairs, registry_for_shot
from atas.providers.wan.journal import JournaledTransport, SubmissionUncertain
def test_reference_order_cannot_swap_characters(tmp_path):
    small=tmp_path/"Small Rabbit.jpg";small.write_bytes(b"small")
    big=tmp_path/"Big Rabbit.jpg";big.write_bytes(b"big")
    shot=SimpleNamespace(entity_ids=("Small Rabbit","Big Rabbit"),character_reference_images=(big,small),raw={},story_id="01")
    assert appearance_pairs(shot)==[("Big Rabbit",big),("Small Rabbit",small)]
    registry=registry_for_shot(shot)
    assert registry[0]["reference_path"]==str(small)
    again=registry_for_shot(SimpleNamespace(**{**vars(shot),"entity_ids":("Big Rabbit","Small Rabbit")}))
    assert registry[0]["entity_id"]==again[1]["entity_id"]


def test_ambiguous_reference_requires_explicit_mapping():
    shot=SimpleNamespace(entity_ids=("A","B"),character_reference_images=(Path("portrait1.png"),Path("portrait2.png")),raw={})
    with pytest.raises(ValueError,match="ambiguous"): appearance_pairs(shot)


def transport(tmp_path,underlying):
    client=SimpleNamespace(config=SimpleNamespace(create_path="/create"),_url=lambda p:p,_task_id=lambda r:r["task_id"])
    request=SimpleNamespace(output_dir=tmp_path,condition=SimpleNamespace(references=[]))
    return JournaledTransport(underlying,client,request)


def test_saved_paid_response_closes_base_state_crash_window(tmp_path):
    calls=[]
    original=SimpleNamespace(request_json=lambda *a,**kw:(calls.append(1) or {"task_id":"paid-task"}))
    payload={"model":"wan3.0-video","input":{"prompt":"scene","media":[]},"parameters":{"duration":4}}
    result=transport(tmp_path,original).request_json("POST","/create",payload=payload)
    repeated=transport(tmp_path,original).request_json("POST","/create",payload=payload)
    assert result==repeated and len(calls)==1


def test_uncertain_post_is_not_repeated(tmp_path):
    calls=[]
    def broken(*a,**kw): calls.append(1);raise TimeoutError("response lost")
    payload={"model":"wan3.0-video","input":{"prompt":"scene"},"parameters":{"duration":4}}
    with pytest.raises(TimeoutError): transport(tmp_path,SimpleNamespace(request_json=broken)).request_json("POST","/create",payload=payload)
    with pytest.raises(SubmissionUncertain): transport(tmp_path,SimpleNamespace(request_json=broken)).request_json("POST","/create",payload=payload)
    assert len(calls)==1


def test_confirmed_capacity_rejection_is_retryable(tmp_path):
    calls=[]
    def busy(*a,**kw):
        calls.append(1)
        if len(calls)==1: raise RuntimeError("HTTP 429 capacity")
        return {"task_id":"accepted"}
    payload={"model":"wan3.0-video","input":{"prompt":"scene"},"parameters":{}}
    wrapped=transport(tmp_path,SimpleNamespace(request_json=busy))
    with pytest.raises(RuntimeError): wrapped.request_json("POST","/create",payload=payload)
    assert wrapped.request_json("POST","/create",payload=payload)["task_id"]=="accepted"


def test_missing_geometry_job_preserves_uncertain_id(monkeypatch):
    from atas.world.geometry_client import GeometryClient, GeometryWaitUncertain
    client=GeometryClient(enabled=True)
    def missing(*a,**kw): raise RuntimeError("geometry HTTP 404: unknown job")
    monkeypatch.setattr(client,"_request",missing)
    with pytest.raises(GeometryWaitUncertain,match="preserving its ID"):
        client.wait("render_original",interval=0,timeout=1)


def test_composition_guide_uses_projected_regions(tmp_path):
    import cv2
    import numpy as np
    from PIL import Image
    from atas.shot.stills import write_composition_guide
    video=tmp_path/"geometry.mp4"
    writer=cv2.VideoWriter(str(video),cv2.VideoWriter_fourcc(*"mp4v"),24,(100,60))
    writer.write(np.zeros((60,100,3),np.uint8));writer.release()
    output=write_composition_guide(video,{"first_frame_subject_regions":[
        {"label":"Rabbit","part":"head","bbox":[.2,.3,.6,.8]}]},tmp_path/"guide.png")
    pixels=np.asarray(Image.open(output))
    assert tuple(pixels[30,20])==(0,255,0)
    raw=write_composition_guide(video,{},tmp_path/"no_boxes.png")
    assert np.asarray(Image.open(raw)).max()==0
    outside=write_composition_guide(video,{'first_frame_subject_regions':[
        {'label':'offscreen','bbox':[2,2,3,3]}, {'bbox':[0,0,0,0]}]},tmp_path/'outside.png')
    assert np.asarray(Image.open(outside)).max()==0






def test_task_the_provider_failed_may_be_submitted_once_more(tmp_path):
    calls=[]
    original=SimpleNamespace(request_json=lambda *a,**kw:(calls.append(1) or {"task_id":f"task-{len(calls)}"}))
    payload={"model":"wan3.0-video","input":{"prompt":"scene","media":[]},"parameters":{"duration":4}}
    first=transport(tmp_path,original).request_json("POST","/create",payload=payload)
    (tmp_path/"wan27"/"run").mkdir(parents=True)
    (tmp_path/"wan27"/"run"/f"state.failed.{first['task_id']}.json").write_text("{}")   # archived by the client
    second=transport(tmp_path,original).request_json("POST","/create",payload=payload)
    assert second["task_id"]!=first["task_id"] and len(calls)==2
    assert transport(tmp_path,original).request_json("POST","/create",payload=payload)==second
