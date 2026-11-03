import threading
from atas_geometry.service import gpu_pool as g
def test_free_vram_does_not_lease_an_externally_busy_card(monkeypatch):
    monkeypatch.setattr(g,'query_gpus',lambda:[g.GPUInfo(i,'test',32000,30000,99 if i in (1,3) else 0) for i in (1,3,5)])
    pool=g.GpuPool(configured=(1,3,5))
    assert pool.acquire(min_free_mib=2048,stop=threading.Event())==5
    pool.release(5)
    assert pool.snapshot()['busy']==[]


def test_available_cards_keep_round_robin(monkeypatch):
    monkeypatch.setattr(g,'query_gpus',lambda:[g.GPUInfo(i,'test',32000,30000,0) for i in (1,3,5)])
    pool=g.GpuPool(configured=(1,3,5))
    chosen=[]
    for _ in range(3):
        index=pool.acquire(stop=threading.Event());chosen.append(index);pool.release(index)
    assert chosen==[1,3,5]
