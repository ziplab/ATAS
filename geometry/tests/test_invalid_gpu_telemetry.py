from types import SimpleNamespace
import threading
from atas_geometry.service import gpu_pool as g
def test_invalid_card_does_not_disable_healthy_cards(monkeypatch):
    rows='3, RTX 5090, 32607, 32109, 0\n6, RTX 5090, 32607, 32109, [N/A]\n7, RTX 5090, 32607, 32109, 0\n'
    monkeypatch.setattr(g.subprocess,'run',lambda *a,**kw:SimpleNamespace(stdout=rows))
    info=g.query_gpus()
    assert len(info)==3 and not info[1].telemetry_valid
    pool=g.GpuPool(configured=(6,3,7))
    assert pool.acquire(stop=threading.Event())==3
    pool.release(3)
    assert pool.acquire(stop=threading.Event())==7
    assert not pool.snapshot()['cards'][0]['telemetry_valid']

def test_invalid_memory_cannot_be_leased(monkeypatch):
    rows='3, RTX 5090, 32607, -1, 0\n4, RTX 5090, 32607, 32109, 0\n'
    monkeypatch.setattr(g.subprocess,'run',lambda *a,**kw:SimpleNamespace(stdout=rows))
    pool=g.GpuPool(configured=(3,4))
    assert pool.acquire(stop=threading.Event())==4
