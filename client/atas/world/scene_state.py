"""Scene presence is persistent; shot focus and camera visibility are not presence."""
from __future__ import annotations
import copy
import json
from dataclasses import replace
from pathlib import Path
from atas.world.identity import appearance_pairs
from atas.request import COMPILER_VERSION
CONTRACT='scene-presence-v1'


def context(shot):
    return getattr(shot,'raw',{}).get('_scene_context',{})


def catalog(plan):
    result=dict(getattr(plan,'payload',{}).get('character_reference_map',{}))
    for shot in plan.shots:
        for label,path in appearance_pairs(shot):
            result.setdefault(label,str(path))
    return result


def prepare(shot,previous,references,*,recapture_state=None):
    state=copy.deepcopy(recapture_state if recapture_state is not None else previous or {})
    entities=state.get('entities',{})
    # The focal list describes intended subjects, not an exclusive room population.
    present={k:v for k,v in entities.items() if v.get('presence')!='departed'}
    # The recursive compiler's onstage list is the screenplay's own statement of who
    # is in the scene for this shot (including people at the frame edge). A character
    # it no longer lists has left by script, even if no exit was filmed.
    left_by_script=[]
    if (shot.raw.get('planning_provenance',{}).get('kind')==COMPILER_VERSION
            and isinstance(shot.raw.get('onstage_labels'),list)):
        onstage=set(shot.raw['onstage_labels'])|set(shot.entity_ids)
        left_by_script=sorted(k for k in present if k not in onstage)
        present={k:v for k,v in present.items() if k in onstage}
    for label in dict.fromkeys([*shot.entity_ids,*shot.raw.get('onstage_labels',[])]):
        present.setdefault(label,{'label':label,'presence':'planned','source_shot_id':None})
    data={'contract':CONTRACT,'scene_id':shot.scene_id,'world_time':float(shot.start),
        'focal_labels':list(shot.entity_ids),'entities':present,
        'reference_paths':{k:v for k,v in references.items() if k in present},
        'source_world_version':state.get('world_version'),'visibility':{},
        'event_snapshot':recapture_state is not None,
        'event_source_time':state.get('event_source_time'),
        'event_source_shot_id':state.get('event_source_shot_id'),
        'left_by_script':left_by_script,
        'policy':'absence_from_frame_is_not_departure; positions_are_last_observed_not_frozen'}
    return replace(shot,raw={**shot.raw,'_scene_context':data})


def brief(shot):
    data=context(shot)
    if not data:return ''
    rows=[]; projections={r['label']:r for r in data.get('visibility',{}).get('actors',[])}
    for label,row in data['entities'].items():
        view=projections.get(label,{})
        rows.append({'character':label,'presence':row['presence'],
            'last_observed_location':row.get('location','unknown'),'last_observed_shot':row.get('source_shot_id'),
            'last_observed_time':row.get('last_observed_time'),
            'view_assessment':view.get('assessment','unknown'),
            'projected_regions':view.get('samples',[])})
    return ('Scene continuity facts (not an instruction to make a group shot): '+json.dumps(rows,ensure_ascii=False)+
        '\nPrimary framing subjects: '+', '.join(data['focal_labels'])+'. '+
        'Other characters remain in the scene unless the story explicitly moves them or shows their departure. '
        'Keep their last supported positions unless this shot specifies movement. Characters projected into view '
        'should appear at those positions when unoccluded; out-of-view characters remain offscreen. '
        'Do not move the camera just to include everyone. Projection is a last-observed pose hypothesis, '
        'not an animation to replay. Unknown visibility must not be interpreted as absence. '
        'Do not introduce unrelated characters.')


def cast_shot(shot,labels=None):
    """Expand appearance/segmentation candidates without changing the focal list."""
    data=context(shot)
    if not data:return shot
    names=list(labels) if labels is not None else list(data['entities'])
    known=set(data['entities'])
    if not set(names)<=known:raise ValueError('unknown scene character')
    paths=data.get('reference_paths',{})
    refs=tuple(Path(paths[name]) for name in names if name in paths)
    return replace(shot,entity_ids=tuple(names),character_reference_images=refs,
        raw={**shot.raw,'character_reference_map':{name:paths[name] for name in names if name in paths}})


def visible_cast(shot,review):
    if not context(shot):return shot
    rows=review.get('scene_observations')
    if not context(shot).get('entities') and rows is None:rows=[]
    if rows is None:return shot
    # Unknown identity is not a geometry veto. Segmentation uses observed classes;
    # only positively observed named characters enter the textual scene state.
    names=[r['label'] for r in rows if r['visibility']=='visible']
    return cast_shot(shot,names)


def validate_observations(rows,shot):
    known=set(context(shot).get('entities',{}))
    if not isinstance(rows,list) or not all(isinstance(r,dict) for r in rows):
        raise ValueError('scene review must address every known character, including offscreen ones')
    # A row for someone outside the scene registry (an extra, or a name the reviewer made
    # up for a person it saw) cannot update the textual state; whether that person belongs
    # in the shot is the film checks' concern, not the registry's.
    rows=[r for r in rows if r.get('label') in known]
    if {r.get('label') for r in rows}!=known or len(rows)!=len(known):
        raise ValueError('scene review must address every known character, including offscreen ones')
    for row in rows:
        missing=[k for k,ok in (('visibility',row.get('visibility') in {'visible','not_visible','uncertain'}),
            ('confidence',isinstance(row.get('confidence'),(int,float)) and 0<=row['confidence']<=1),
            ('evidence',bool(row.get('evidence')))) if not ok]
        if missing:
            raise ValueError(f'scene_observations row {row["label"]!r} needs valid fields: {", ".join(missing)} '
                             '(evidence is a separate non-empty frame/time string, not part of location)')
        if row.get('departure_observed') and (row['confidence']<.9 or not row.get('departure_evidence')):
            raise ValueError('departure needs explicit visible evidence, not missing pixels')
        if not isinstance(row.get('departure_observed',False),bool):
            raise ValueError('departure_observed must be a boolean')
    return rows


def advance(shot,previous,review,identity,world_version):
    """Textual film observations, not certified persistent 3D identities."""
    from atas.world.entity_graph import EntityGraph
    state=copy.deepcopy(previous or {'entities':{}})
    observations={r['label']:r for r in review.get('scene_observations',[])}
    # Appearance registry IDs are labels, not proof of a SAM track association.
    # Preserve the legacy argument for callers; normal execution needs no binding.
    from atas.world.identity import registry_for_shot
    observed_bindings={r['entity_id']:{'label':r['label']} for r in registry_for_shot(shot)
        if observations.get(r['label'],{}).get('visibility')=='visible'}
    identity={'bindings':{**identity.get('bindings',{}),**observed_bindings}}
    for eid,binding in identity.get('bindings',{}).items():
        label=binding['label'];obs=observations.get(label,{})
        state['entities'][label]={'label':label,'entity_id':eid,'presence':'present',
            'source_shot_id':shot.shot_id,'last_observed_time':float(shot.start+shot.duration),
            'location':obs.get('location') if obs.get('location') not in (None,'','unknown') else state['entities'].get(label,{}).get('location','unknown'),
            'evidence':obs.get('evidence','film observation'),
            'identity_verified':False,'observation_basis':'film_vlm_not_sam_identity',
            'reference_path':context(shot).get('reference_paths',{}).get(label)}
    for label,obs in observations.items():
        if label in state['entities'] and obs.get('departure_observed') and obs.get('confidence',0)>=.9:
            state['entities'][label].update(presence='departed',departure_shot_id=shot.shot_id,
                departure_evidence=obs['departure_evidence'])
    state.update(schema_version=1,scene_id=shot.scene_id,world_version=world_version,
        film_audit_decision=review.get('decision'),
        scripted_event_verified=review.get('decision') in {'pass','warning'} and all(
            any(c.get('criterion')==name and c.get('status')=='pass' for c in review.get('checks',[]))
            for name in ('action','event')),
        world_time=float(shot.start+shot.duration),accepted_shot_id=shot.shot_id,
        observed_shot_id=shot.shot_id,observation_basis='film_review_not_verified_3d_tracks')
    nodes=[{'node_id':shot.scene_id,'type':'scene'}];relations=[]
    for row in state['entities'].values():
        nodes.append({'node_id':row['entity_id'],'type':'entity','label':row['label']})
        relations.append({'relation_id':row['entity_id']+':presence','subject_id':row['entity_id'],
            'predicate':'present_in' if row['presence']=='present' else 'departed_from','object_id':shot.scene_id,
            'source_shot_id':row['source_shot_id'],'last_observed_time':row['last_observed_time'],
            'location_description':row.get('location'),'position_semantics':'last_observed_not_permanent'})
    graph=EntityGraph(nodes=tuple(nodes),relations=tuple(relations))
    if graph.closure_errors():raise ValueError('invalid entity scene graph')
    state['graph']=graph.to_json_dict()
    return state


def event_snapshot(before,after,source_shot_id,source_time):
    """Event membership comes from that event, never from a later scene visit."""
    result=copy.deepcopy(before or {'entities':{}})
    result.setdefault('entities',{})
    for label,row in after.get('entities',{}).items():
        if row.get('source_shot_id')==source_shot_id:
            result['entities'][label]={**row,'presence':'event_member','location':'consult source event frame; end pose not reused'}
    result['event_source_time']=float(source_time)
    result['event_source_shot_id']=source_shot_id
    return result
