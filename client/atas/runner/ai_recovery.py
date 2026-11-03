"""Persistent, bounded AI correction; never changes acceptance criteria."""
import json,re,time
from pathlib import Path
from atas.runner.recon_ledger import write_json_atomic
class AIRecoveryExhausted(ValueError):
    pass

def safe_error(exc):
    text=re.sub(r'https?://\S+','<url>',str(exc))
    text=re.sub(r'(?i)(bearer\s+|api[_-]?key[=: ]+)[^\s,;]+',r'\1<redacted>',text)
    return text[:1800]

def corrected_json(folder,key,generate,validate,*,feedback=None,max_attempts=3):
    """generate(history) -> text; validate(text) -> validated object.

    Only malformed/semantically rejected responses are reauthored. Transport
    errors propagate to operational recovery. Started calls consume a slot on
    restart; paid image/video submissions must never use this text-only helper.
    """
    root=Path(folder)/key;root.mkdir(parents=True,exist_ok=True)
    history=[]
    for index in range(max_attempts):
        path=root/f'attempt_{index:02d}.json'
        if path.exists():
            row=json.loads(path.read_text())
            if row.get('status') == 'evidence_exhausted':
                raise AIRecoveryExhausted(row.get('error','evidence exhausted'))
            if row.get('status')=='valid' and feedback is None:return validate(row['response'])
            if row.get('status')=='invalid' and feedback is None and row.get('response') is not None:
                # A validator fixed since this attempt may accept the stored response.
                try:result=validate(row['response'])
                except (ValueError,KeyError,TypeError,IndexError):pass
                else:
                    row.update(status='valid',revalidated_at=time.time());write_json_atomic(path,row)
                    return result
            history.append(row);continue
        row={'index':index,'status':'started','started_at':time.time()}
        write_json_atomic(path,row)
        try:
            response=generate({'attempt':index,'previous_attempts':history,'execution_feedback':feedback,
                'instruction':'Diagnose the exact failure and return a corrected complete response. Preserve screenplay, explicit motion, world facts and all validation requirements. Never mark unobserved actors as observed or remove required actors to pass.'})
        except Exception as exc:
            row.update(status='call_failed',error=safe_error(exc));write_json_atomic(path,row);raise
        row['response']=response
        try:result=validate(response)
        except (ValueError,KeyError,TypeError,IndexError) as exc:
            row.update(status='invalid',error=safe_error(exc),finished_at=time.time())
            if isinstance(exc, AIRecoveryExhausted):row['status']='evidence_exhausted'
            write_json_atomic(path,row);history.append(row)
            if isinstance(exc, AIRecoveryExhausted):
                # Explicit evidence exhaustion is not malformed JSON. Repeating
                # the same observations cannot reveal a missing subject.
                raise
            continue
        row.update(status='valid',finished_at=time.time());write_json_atomic(path,row)
        return result
    raise AIRecoveryExhausted('AI correction budget exhausted; inspect '+str(root))

CONTENT_POLICY_MARKERS=('moderation_blocked','facedetectionsuspect','real human faces',
    'rejected by the safety system','data_inspection_failed','content_policy_violation')


def content_policy_refusal(error):
    """A provider refused this input on content grounds, not a pipeline defect.

    Screenplay sources may legitimately contain real photographic people, so these
    shots are unproducible here. Report them apart from genuine failures instead
    of retrying, rewriting around a safety system, or counting them as defects.
    """
    text=str(error).lower()
    return any(marker in text for marker in CONTENT_POLICY_MARKERS)


def camera_failure_recoverable(error):
    text=str(error).lower()
    return any(s in text for s in ('camera context changed','no reliable seed depth',
        'ambiguous seed depth','insufficient grounded region','no geometrically safe same-intent',
        'subject visually unresolved','subject seed is outside','subject region'))
