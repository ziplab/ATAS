import asyncio,json
from dataclasses import dataclass,field
import pytest
from atas.review.repair import repair_until_reviewed, promote_reviewed
from atas.review.film_review import file_hash
@dataclass
class Shot:
    prompt:str='A close shot in the same room'
    raw:dict=field(default_factory=dict)

def review(shot,path,directory,*,stage,**kwargs):
    failed=path.read_bytes()==b'bad'
    return {'stage':stage,'decision':'hold' if failed else 'pass','summary':'material drift' if failed else 'ok',
        'video_sha256':file_hash(path),'checks':[{'criterion':'world_consistency',
        'status':'fail' if failed else 'pass','severity':'critical','confidence':.99,'evidence':'known blanket changed'}]}

def test_failed_film_revised_once_then_replayed_without_generation(tmp_path):
    source=tmp_path/'result.mp4';source.write_bytes(b'bad');calls=[]
    async def revise(shot,d,feedback,failed):
        calls.append(feedback);p=d/'result.mp4';p.write_bytes(b'good');return shot,p
    async def run():return await repair_until_reviewed(Shot(),source,tmp_path,stage='film',review=review,revise=revise)
    _,candidate,r=asyncio.run(run());assert r['decision']=='pass' and source.read_bytes()==b'bad'
    (tmp_path/'geometry_recon_result.json').write_text('{}')
    promote_reviewed(candidate,source,stage='film',shot_dir=tmp_path)
    assert source.read_bytes()==b'good' and not (tmp_path/'geometry_recon_result.json').exists()
    asyncio.run(run());assert len(calls)==1
    assert list((tmp_path/'repairs/superseded').glob('*/result.mp4'))[0].read_bytes()==b'bad'

def test_exhausted_budget_survives_resume(tmp_path):
    source=tmp_path/'result.mp4';source.write_bytes(b'bad');calls=[]
    async def revise(shot,d,*a):
        calls.append(1);p=d/'result.mp4';p.write_bytes(b'bad');return shot,p
    for _ in range(2):
        with pytest.raises(RuntimeError):asyncio.run(repair_until_reviewed(Shot(),source,tmp_path,
            stage='film',review=review,revise=revise,max_repairs=2))
    assert len(calls)==2 and source.read_bytes()==b'bad'
    assert json.loads((tmp_path/'repair_required.json').read_text())['budget_exhausted']
    entries=json.loads((tmp_path/'repairs/film/ledger.json').read_text())['attempts']
    assert entries[1]['feedback']['repair_history'][0]['checks'][0]['criterion']=='world_consistency'

@pytest.mark.parametrize('guard',['world_publication.json','scene_state_after.json','geometry_recon_state.json'])
def test_no_regeneration_of_accepted_or_inflight_shot(tmp_path,guard):
    source=tmp_path/'result.mp4';source.write_bytes(b'bad')
    (tmp_path/guard).write_text(json.dumps({'status':'waiting'}))
    async def revise(*a):raise AssertionError('must not generate')
    with pytest.raises(RuntimeError):asyncio.run(repair_until_reviewed(Shot(),source,tmp_path,
        stage='film',review=review,revise=revise))

def test_pending_attempt_resumes_same_directory(tmp_path):
    source=tmp_path/'result.mp4';source.write_bytes(b'bad');dirs=[]
    async def revise(shot,d,*a):
        dirs.append(d);p=d/'result.mp4'
        if not p.exists():p.write_bytes(b'good');raise ConnectionError('lost polling reply')
        return shot,p
    with pytest.raises(ConnectionError):asyncio.run(repair_until_reviewed(Shot(),source,tmp_path,
        stage='film',review=review,revise=revise))
    asyncio.run(repair_until_reviewed(Shot(),source,tmp_path,stage='film',review=review,revise=revise))
    assert dirs[0]==dirs[1]
    assert len(json.loads((tmp_path/'repairs/film/ledger.json').read_text())['attempts'])==1

def test_warning_does_not_spend_generation_budget(tmp_path):
    p=tmp_path/'result.mp4';p.write_bytes(b'good')
    def uncertain(*a,**kw):return {**review(*a,**kw),'decision':'warning'}
    async def revise(*a):raise AssertionError('not a failure')
    _,_,r=asyncio.run(repair_until_reviewed(Shot(),p,tmp_path,stage='film',review=uncertain,revise=revise))
    assert r['decision']=='warning'
