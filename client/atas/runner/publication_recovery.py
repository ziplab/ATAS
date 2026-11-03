"""Bounded world-only recovery; never generates or changes a film."""
import json
from atas.runner.recon_ledger import record_succeeded, write_json_atomic
def publish_with_rebase(geometry, shot_dir, current, payload):
    receipt = shot_dir / 'world_rebase.json'
    state = json.loads(receipt.read_text()) if receipt.exists() else {'attempts': []}
    attempts = state['attempts']
    while True:
        pending = next((a for a in attempts if a.get('status') == 'waiting'), None)
        if pending:
            # Polling timeout leaves this ID durable for world-only resume.
            current = geometry.wait(pending['job_id'])
            record_succeeded(shot_dir, current)
            pending['status'] = 'completed'
            write_json_atomic(receipt, state)
        fuse = current['fuse']
        if fuse.get('registered') is not True:
            raise RuntimeError('rebased reconstruction rejected geometry; film retained, old world unchanged')
        if fuse.get('publication_status') == 'committed':
            return current, fuse
        request = {**payload, 'candidate_job_id': fuse['candidate_job_id']}
        try:
            publication = geometry._request('POST', '/v1/world-publication', request)
            return current, publication
        except Exception as exc:
            # Other 409s are provenance/lineage errors, NOT permission to rebase.
            if 'world changed before publication; candidate must be rebuilt' not in str(exc):
                raise
            if len(attempts) >= 2:
                raise RuntimeError('world rebase budget exhausted; film retained, world pending manual reconciliation') from exc
        response = geometry._request('POST', '/v1/world-rebase', request)
        if not response.get('job_id'):
            raise RuntimeError('world rebase returned no job ID')
        attempts.append({**response, 'status': 'waiting'})
        write_json_atomic(receipt, state)
