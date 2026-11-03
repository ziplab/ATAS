"""Bounded registration using distinct trusted references; never relax gates."""
import json
import time
from pathlib import Path
from atas_geometry.service.durable import atomic_json, file_hash
from atas_geometry.service.resource_errors import ResourceUnavailableError
POLICY = 'distinct_trusted_reference_recovery_v1'
MAX_ATTEMPTS = 2


def accepted(report):
    return (report.get('reconstruction', {}).get('geometry_accepted') is True
            and report.get('cross_shot_validation', {}).get('accepted') is True)


def diagnose(report):
    recon = report.get('reconstruction', {})
    tolerance = recon.get('anchor_tolerance')
    failures = []
    for chunk in recon.get('chunks', []):
        reasons = []
        if tolerance is not None:
            if chunk.get('heldout_median', float('inf')) > tolerance:
                reasons.append('heldout_history_median')
            if chunk.get('heldout_within_2t', 0) < .65:
                reasons.append('heldout_history_support')
            if chunk.get('seam_median') is not None and chunk['seam_median'] > 2*tolerance:
                reasons.append('chunk_seam')
        if chunk.get('pinhole_residual_pixels', 0) > 2:
            reasons.append('pinhole_fit')
        if not chunk.get('accepted_geometry'):
            row = {'chunk': chunk.get('chunk'), 'checks': reasons or ['unspecified_geometry_check']}
            if tolerance:
                # Measured value as a multiple of its limit (> 1 fails).
                if chunk.get('heldout_median') is not None:
                    row['heldout_x_limit'] = round(chunk['heldout_median'] / tolerance, 2)
                if chunk.get('seam_median') is not None:
                    row['seam_x_limit'] = round(chunk['seam_median'] / (2 * tolerance), 2)
                frames = [f for f in chunk.get('heldout_frames', []) if f.get('median') is not None]
                if frames:
                    row['heldout_frames_x_limit'] = {str(f['frame']): round(f['median'] / tolerance, 2) for f in frames}
            failures.append(row)
    return {'historical_reconstruction_accepted': recon.get('geometry_accepted'),
            'cross_shot_accepted': report.get('cross_shot_validation', {}).get('accepted'),
            'cross_shot_status': report.get('cross_shot_validation', {}).get('status', 'measured'),
            'rejection_category': ('internal_reconstruction' if recon.get('geometry_accepted') is False
                else 'cross_shot_alignment' if report.get('cross_shot_validation', {}).get('accepted') is False
                else None),
            'failed_chunks': failures,
            'chunks_total': len(recon.get('chunks', [])),
            'cross_shot': {k: report.get('cross_shot_validation', {}).get(k) for k in
                           ('correspondences', 'frame_pairs', 'within_2t_fraction')} | {
                'median_x_limit': (round(report['cross_shot_validation']['median_world_error']
                                         / report['cross_shot_validation']['tolerance_world_units'], 2)
                                   if report.get('cross_shot_validation', {}).get('median_world_error') is not None
                                   and report['cross_shot_validation'].get('tolerance_world_units') else None)},
            'reference_l0': recon.get('reference_l0'),
            'interpretation': 'history self-reconstruction and new/old matching are different checks; '
                              'a failed history check alone does not identify generated-video error',
            'acceptance_rules_changed': False}


def run_bounded(folder, *, stamp, references, execute, recover_interrupted=False):
    """execute(reference, attempt_dir) writes a fusion_report.json.

    A started attempt without its final report is uncertain, never resubmitted
    implicitly. Distinct references must already be trusted by the caller.
    """
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    refs = list(dict.fromkeys(str(Path(p).resolve()) for p in references))[:MAX_ATTEMPTS]
    if not refs: raise ValueError('no trusted registration reference')
    path = folder/'registration_recovery.json'
    contract = {'policy': POLICY, 'stamp': stamp, 'references': refs, 'max_attempts': MAX_ATTEMPTS}
    ledger = json.loads(path.read_text()) if path.exists() else {'contract': contract, 'attempts': []}
    if ledger['contract'] != contract: raise ValueError('registration recovery inputs changed; explicit revision required')
    def execute_attempt(reference, attempt, row):
        try:
            execute(reference, attempt)
        except ResourceUnavailableError as exc:
            # This reference has not received a geometry verdict. Do not spend
            # the bounded unknown-interruption/semantic registration allowance.
            # The supervisor/resource queue owns the separate retry budget.
            row.update(status='waiting_resource', resource_interruptions=int(row.get('resource_interruptions', 0))+1,
                       last_resource_failure={**exc.fields(), 'error': str(exc), 'recorded_at': time.time()})
            ledger['status'] = 'waiting_resource'
            atomic_json(path, ledger)
            raise
    for index, ref in enumerate(refs):
        attempt = folder/f'attempt_{index:03d}'
        report_path = attempt/'fusion_report.json'
        # Adopt the original completed attempt without recomputing it. Its
        # stored reference must match, so another scene cannot be reused.
        if index >= len(ledger['attempts']):
            partial = attempt.exists() and not report_path.exists()
            ledger['attempts'].append({'reference': ref, 'directory': str(attempt), 'status': 'started'})
            atomic_json(path, ledger)
            if partial:
                if not recover_interrupted:
                    raise RuntimeError('registration attempt incomplete; reconcile before retry')
                # Supervisor has already terminated and joined the prior owned
                # worker. Retain all unpublished bytes, never overwrite them.
                attempt.rename(folder/f'interrupted_{index:03d}_{time.time_ns()}')
                ledger['attempts'][index]['interrupted_retries'] = 1
                atomic_json(path, ledger)
            if not report_path.exists(): execute_attempt(Path(ref), attempt, ledger['attempts'][index])
        row = ledger['attempts'][index]
        if not report_path.exists():
            if not recover_interrupted:
                raise RuntimeError('registration attempt incomplete; reconcile before retry')
            resource_interrupted = row.get('status') == 'waiting_resource'
            tries = int(row.get('interrupted_retries', 0))
            if not resource_interrupted and tries >= 1:
                raise RuntimeError('interrupted registration recovery budget exhausted')
            if attempt.exists():
                prefix = 'resource_interrupted' if resource_interrupted else 'interrupted'
                attempt.rename(folder/f'{prefix}_{index:03d}_{time.time_ns()}')
            if resource_interrupted:
                row.update(status='restarted_after_resource_readmission')
            else:
                row.update(interrupted_retries=tries+1, status='restarted_after_owned_worker_exit')
            atomic_json(path, ledger)
            execute_attempt(Path(ref), attempt, row)
            if not report_path.exists():
                raise RuntimeError('registration worker returned without final report')
        report = json.loads(report_path.read_text())
        actual_ref = report.get('reference_l0') or report.get('reconstruction', {}).get('reference_l0')
        if not actual_ref or str(Path(actual_ref).resolve()) != ref:
            raise ValueError('registration report reference mismatch')
        if 'source_l0' in stamp:
            actual_source = report.get('source_l0')
            if not actual_source or str(Path(actual_source).resolve()) != stamp['source_l0']:
                raise ValueError('registration report source mismatch')
        if 'l1_sha256' in stamp and report.get('original_l1_sha256') != stamp['l1_sha256']:
            raise ValueError('registration report parent world mismatch')
        digest = file_hash(report_path)
        if row.get('report_sha256') and row['report_sha256'] != digest:
            raise ValueError('completed registration report changed')
        row.update(status='accepted' if accepted(report) else 'rejected',
                   report_sha256=digest, diagnosis=diagnose(report))
        ledger['status'] = 'accepted' if accepted(report) else 'trying_alternative_reference'
        atomic_json(path, ledger)
        if accepted(report): return report, ledger
    ledger['status'] = 'exhausted'
    ledger['reason'] = 'no_untried_trusted_reference' if len(refs) < MAX_ATTEMPTS else 'registration_repair_budget_exhausted'
    atomic_json(path, ledger)
    return report, ledger
