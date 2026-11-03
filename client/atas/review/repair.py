"""Bounded, durable visual repair. Never bypass geometry or edit accepted worlds."""
import hashlib
import json
import shutil
import time
from contextlib import contextmanager
from pathlib import Path
from dataclasses import replace
from atas.review.film_review import file_hash
from atas.runner.recon_ledger import write_json_atomic
@contextmanager
def repair_phase(folder, phase, attempt):
    """Timing only; never changes the candidate or its acceptance authority."""
    path=folder/'timings.json'
    data=json.loads(path.read_text()) if path.exists() else {'attempts':[]}
    item={'phase':phase,'attempt':attempt,'started_at':time.time()}
    data['active']=item;write_json_atomic(path,data)
    tick=time.perf_counter()
    try:
        yield
    except BaseException as exc:
        item.update(outcome='failed',error_type=type(exc).__name__)
        raise
    else:item['outcome']='succeeded'
    finally:
        item.update(seconds=time.perf_counter()-tick,finished_at=time.time())
        data.pop('active',None);data['attempts'].append(item);write_json_atomic(path,data)


class RepairExhausted(RuntimeError):
    """Definite visual rejection after the durable generation budget is spent."""
    def __init__(self, review):
        self.review = review
        evidence='; '.join(str(c.get('evidence','')) for c in review.get('checks',[]) if c.get('status')=='fail')
        super().__init__('visual repair budget exhausted: ' + str(review.get('summary', '')) + '; ' + evidence)


def assert_repairable(shot_dir):
    if any((shot_dir/n).exists() for n in ('world_publication.json','scene_state_after.json')):
        raise RuntimeError('accepted observation is immutable; automatic regeneration refused')
    state=shot_dir/'geometry_recon_state.json'
    if state.is_file() and json.loads(state.read_text()).get('status') in ('submitted','waiting'):
        raise RuntimeError('reconstruction may still be running; resolve it before regenerating')


def promote_reviewed(source, destination, *, stage, shot_dir):
    """Keep the old artifact and its reconstruction receipts recoverable."""
    source,destination=Path(source),Path(destination)
    if source.resolve()==destination.resolve():return destination
    if destination.is_file() and file_hash(source)==file_hash(destination):return destination
    assert_repairable(shot_dir)
    digest=file_hash(destination) if destination.is_file() else 'none'
    archive=shot_dir/'repairs'/'superseded'/digest
    archive.mkdir(parents=True,exist_ok=True)
    if destination.is_file() and not (archive/destination.name).exists():
        shutil.copy2(destination,archive/destination.name)
    # Move only unaccepted reconstruction receipts. A new film MUST NOT reuse
    # an old success, failure, keyword intent or identity binding.
    if stage=='film':
        for name in ('geometry_recon_intent.json','geometry_recon_result.json','geometry_recon_state.json',
                     'identity_binding.json','identity_bindings.json','dynamic_keywords.json'):
            path=shot_dir/name
            if path.exists():
                target=archive/name
                if target.exists():
                    if file_hash(target)!=file_hash(path):raise RuntimeError('superseded receipt collision')
                else:shutil.copy2(path,target)
                path.unlink()
    temporary=destination.with_name(destination.name+'.reviewed_pending')
    shutil.copy2(source,temporary);temporary.replace(destination)
    ledger=shot_dir/'repairs'/stage/'ledger.json'
    if ledger.exists():
        state=json.loads(ledger.read_text())
        selected=next((e for e in state['attempts'] if e.get('sha256')==file_hash(source)),None)
        if selected:
            selected_dir=Path(selected['directory'])
            names=['generation_prompts.json']+(['wan_lane.json'] if stage=='film' else [])
            for name in names:
                src=selected_dir/name;dst=shot_dir/name
                if not src.is_file():continue
                if dst.is_file() and not (archive/name).exists():shutil.copy2(dst,archive/name)
                temp=dst.with_name(name+'.reviewed_pending');shutil.copy2(src,temp);temp.replace(dst)
            write_json_atomic(shot_dir/'generation_repair_status.json',{'stage':stage,'status':'reviewed',
                'selected_attempt':str(selected_dir),'selected_sha256':file_hash(source),
                'decision':selected['review']['decision'],'attempts_used':len(state['attempts']),
                'world_publication_not_implied':True})
    return destination


async def repair_until_reviewed(shot,artifact,shot_dir,*,stage,review,revise,
                                review_kwargs=None,max_repairs=1,context_key=None):
    """revise(shot, attempt_dir, feedback, failed_media) -> (shot, media).

    Each attempt is journaled BEFORE generating, and its provider output lives
    in that same attempt directory. Resume calls revise on a pending attempt,
    whose author/provider caches resume the same work rather than creating a
    new attempt. No retries for mere uncertainty or numerical camera mismatch.
    """
    if stage not in ('first_frame','film'):raise ValueError('invalid repair stage')
    if not 0<=max_repairs<=(4 if stage=='film' else 2):
        raise ValueError('film allows four repairs; opening allows two')
    artifact=Path(artifact);kwargs=dict(review_kwargs or {})
    key=hashlib.sha256(json.dumps({'script':shot.prompt,'context':context_key,
        'scene':getattr(shot,'raw',{}).get('_scene_context',{}),'stage':stage},
        sort_keys=True,default=str).encode()).hexdigest()
    folder=shot_dir/'repairs'/stage;folder.mkdir(parents=True,exist_ok=True)
    ledger=folder/'ledger.json'
    state=json.loads(ledger.read_text()) if ledger.is_file() else {
        'schema_version':1,'context_key':key,'budget':max_repairs,'attempts':[]}
    if state['context_key']!=key:
        # Inputs changed since the ledger was written (e.g. a re-planned camera).
        # Keep the old attempts for inspection and start a fresh ledger.
        archive=folder.with_name(stage+'_superseded_'+state['context_key'][:12])
        if not archive.exists():folder.rename(archive)
        folder.mkdir(parents=True,exist_ok=True)
        state={'schema_version':1,'context_key':key,'budget':max_repairs,'attempts':[]}
    budget=min(max_repairs,state['budget'])
    current=shot
    # Initial review remains at the canonical path. It is content/policy keyed.
    with repair_phase(folder,'review',0):
        initial=review(current,artifact,shot_dir,stage=stage,**kwargs)
    if not state['attempts'] and initial['decision']!='hold':return current,artifact,initial
    failed,feedback=artifact,initial
    if state.get('original_sha256') and file_hash(artifact) not in {
        state['original_sha256'],state.get('approved_sha256')}:
        raise RuntimeError('film changed outside its repair ledger')
    for index in range(budget):
        entries=state['attempts']
        if index>=len(entries):
            if feedback['decision']!='hold':break
            assert_repairable(shot_dir)
            # Diagnostic input is immutable even when the canonical result is
            # eventually replaced by an approved revision.
            attempt=folder/f'attempt_{index+1:02d}';attempt.mkdir(exist_ok=True)
            prior=attempt/('failed.png' if stage=='first_frame' else 'failed.mp4')
            if not prior.exists():shutil.copy2(failed,prior)
            cumulative={**feedback,'repair_history':[
                {k:e['feedback'].get(k) for k in ('stage','checks','summary')} for e in entries]}
            entry={'directory':str(attempt),'failed_media':str(prior),'failed_sha256':file_hash(prior),
                'feedback':cumulative,'status':'pending'}
            entries.append(entry);state.setdefault('original_sha256',file_hash(artifact))
            write_json_atomic(ledger,state)
        entry=entries[index];attempt=Path(entry['directory'])
        if entry['status']=='pending':
            assert_repairable(shot_dir)
            if file_hash(entry['failed_media'])!=entry['failed_sha256']:raise RuntimeError('repair evidence changed')
            with repair_phase(folder,'revise_and_generate',index+1):
                current,candidate=await revise(current,attempt,entry['feedback'],Path(entry['failed_media']))
            candidate=Path(candidate)
            with repair_phase(folder,'review',index+1):
                checked=review(current,candidate,attempt,stage=stage,context_dir=shot_dir,**kwargs)
            entry.update(status='reviewed',media=str(candidate),sha256=file_hash(candidate),review=checked,
                authored=getattr(current,'raw',{}).get('_generation_prompts'))
            write_json_atomic(ledger,state)
        else:
            candidate=Path(entry['media'])
            if file_hash(candidate)!=entry['sha256']:raise RuntimeError('repair output changed')
            if entry.get('authored'):current=replace(current,raw={**current.raw,'_generation_prompts':entry['authored']})
            # Re-audit after policy/history changes; cache avoids repeat calls otherwise.
            with repair_phase(folder,'review_cached_or_revalidate',index+1):
                checked=review(current,candidate,attempt,stage=stage,context_dir=shot_dir,**kwargs)
            entry['review']=checked;write_json_atomic(ledger,state)
        if checked['decision']!='hold':
            state.update(approved_sha256=file_hash(candidate),approved_media=str(candidate))
            write_json_atomic(ledger,state)
            return current,candidate,checked
        failed,feedback=candidate,checked
    write_json_atomic(shot_dir/'repair_required.json',{'stage':stage,'budget_exhausted':True,
        'ledger':str(ledger),'review':feedback,'world_writeback_allowed':False})
    raise RepairExhausted(feedback)
