#!/usr/bin/env python3
"""Prepare a corrected-baseline run before transferring a live MORAI trainer.

The old process remains running during metadata checks and baseline evaluation.
At a NEW durable checkpoint it is suspended, then resumed automatically if the
replacement fails before producing a verified first-update checkpoint. The old
process is terminated only after successful model+AdamW restoration and save.
"""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


class HandoffInterrupted(RuntimeError):
    pass


def interrupt_handoff(signum, _frame):
    raise HandoffInterrupted('Handoff interrupted by signal %s' % signum)


def stop_replacement(child):
    """An exit racing with cleanup must not prevent the original from resuming."""
    if child is None or child.poll() is not None:
        return
    try:
        os.killpg(child.pid,signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        child.wait(timeout=15)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(child.pid,signal.SIGKILL)
        except ProcessLookupError:
            pass
        child.wait(timeout=15)


def rollback(child,pid,token,paused,report):
    try:
        stop_replacement(child)
    finally:
        if paused:
            try:
                signal_original(pid,token,signal.SIGCONT)
                report['old_training_resumed']=True
            except (ProcessLookupError,FileNotFoundError):
                report['old_training_resumed']=False
                report['old_process_already_exited']=True


def write_json(path,value):
    partial=path.with_suffix(path.suffix+'.partial')
    partial.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n')
    partial.replace(path)


def sha256(path):
    digest=hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda:source.read(4*1024*1024),b''):digest.update(block)
    return digest.hexdigest()


def process_identity(pid):
    proc=Path('/proc')/str(pid)
    command=proc.joinpath('cmdline').read_bytes().split(b'\x00')
    args=[v.decode() for v in command if v]
    # The comm name may contain spaces; fields after its closing ')' are stable.
    fields=proc.joinpath('stat').read_text().rsplit(')',1)[1].split()
    return args,fields[19]


def find_trainer(output):
    matches=[]
    for path in Path('/proc').iterdir():
        if not path.name.isdigit():continue
        try:args,token=process_identity(int(path.name))
        except (OSError,ValueError):continue
        if ('vision-encoder-morai' in args and '--output' in args
                and args[args.index('--output')+1]==str(output)):
            matches.append((int(path.name),token))
    if len(matches)!=1:raise RuntimeError('Expected one live trainer for '+str(output))
    return matches[0]


def signal_original(pid,token,sig):
    _,current=process_identity(pid)
    if current!=token:raise RuntimeError('PID identity changed; refusing to signal')
    os.kill(pid,sig)


def snapshot_checkpoint(source,destination):
    """Open one immutable inode; publish only after CPU deserialization succeeds."""
    import torch
    temporary=destination.with_suffix(destination.suffix+'.partial')
    hasher=hashlib.sha256()
    with source.open('rb') as reader,temporary.open('xb') as writer:
        stat=os.fstat(reader.fileno())
        for block in iter(lambda:reader.read(4*1024*1024),b''):
            writer.write(block);hasher.update(block)
        writer.flush();os.fsync(writer.fileno())
    checkpoint=torch.load(str(temporary),map_location='cpu',weights_only=True)
    if (not isinstance(checkpoint.get('state_dict'),dict)
            or not checkpoint.get('optimizer',{}).get('state')
            or not isinstance(checkpoint.get('steps'),int)
            or checkpoint['steps']<=0):
        raise ValueError('Checkpoint has no usable weights/optimizer/update count')
    result=dict(steps=checkpoint['steps'],epoch=checkpoint['epoch'],sha256=hasher.hexdigest(),
                source_inode=stat.st_ino,bytes=stat.st_size)
    del checkpoint
    temporary.replace(destination)
    return result


def wait_for(path,child,timeout,phase):
    start=time.monotonic();last_print=0
    while not path.is_file():
        if child.poll() is not None:raise RuntimeError('Replacement exited during '+phase)
        if time.monotonic()-start>timeout:raise TimeoutError(phase+' timed out')
        if time.monotonic()-last_print>=30:
            print(json.dumps(dict(phase=phase,elapsed_seconds=round(time.monotonic()-start))),flush=True)
            last_print=time.monotonic()
        time.sleep(1)
    return json.loads(path.read_text())


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--old-run',type=Path,required=True)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--control-root',type=Path,required=True)
    parser.add_argument('--workers',type=int,default=2)
    parser.add_argument('--prepare-timeout',type=int,default=14400)
    parser.add_argument('--checkpoint-timeout',type=int,default=1800)
    parser.add_argument('--restore-timeout',type=int,default=900)
    parser.add_argument('--dry-run',action='store_true')
    args=parser.parse_args()
    old=args.old_run.resolve();config=args.config.resolve()
    pid,token=find_trainer(old)
    prior=json.loads((old/'config.json').read_text())
    cfg=json.loads(config.read_text())
    if (prior['args']['frame_sampling']!='all' or prior['batch_size']!=1
            or prior['args']['checkpoint_selection']!='scenario_guarded'
            or prior['args']['checkpoint_interval']<=0
            or cfg['checkpoint_sha256']!=prior['initial_checkpoint_sha256']):
        raise ValueError('Handoff only supports the audited all-frame guarded experiment and original anchor')
    stamp=datetime.datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+str(os.getpid())
    control=args.control_root.resolve()/('handoff_'+stamp)
    new=old.parent/('mixed_grid_fast_'+stamp)
    resume=control/'checkpoint-handoff.pth.tar'
    root=Path(__file__).resolve().parents[1]
    command=[sys.executable,'-u',str(root/'train.py'),'vision-encoder-morai',
        '--config',str(config),'--output',str(new),'--initialization','morai_checkpoint',
        '--trainable-scope',prior['trainable_scope'],'--epochs',str(prior['args']['epochs']),
        '--learning-rate',str(prior['args']['learning_rate']),'--frame-sampling','all',
        '--detection-loss-policy',prior['args']['detection_loss_policy'],
        '--velocity-source',prior['velocity_source'],'--checkpoint-selection','scenario_guarded',
        '--early-stop-patience',str(prior['args']['early_stop_patience']),
        '--workers',str(args.workers),'--validation-workers',str(args.workers),
        '--threads',str(prior['args']['threads']),'--seed',str(prior['args']['seed']),
        '--checkpoint-interval',str(prior['args']['checkpoint_interval']),
        '--log-interval',str(prior['args']['log_interval']),
        '--pin-memory','--prefetch-factor','2','--resume-from',str(resume),
        '--wait-for-resume','--allow-legacy-sampler-reset']
    if args.dry_run:
        print(json.dumps(dict(old_pid=pid,new_output=str(new),command=command),indent=2));return
    control.mkdir(parents=True,exist_ok=False)
    log_path=control/'train.log'
    report=dict(status='preparing',old_run=str(old),old_pid=pid,old_pid_start=token,
                output=str(new),log=str(log_path),command=command)
    write_json(control/'handoff.json',report)
    print('새 실행 로그: '+str(log_path),flush=True)
    print('기존 학습은 새 기준 평가가 완료될 때까지 계속 실행됩니다.',flush=True)
    env=dict(os.environ,PYTHONNOUSERSITE='1')
    paused=False;confirmed=False;child=None
    previous_handlers={sig:signal.signal(sig,interrupt_handoff)
                       for sig in (signal.SIGINT,signal.SIGTERM)}
    try:
        with log_path.open('xb') as log:
            child=subprocess.Popen(command,cwd=str(root),env=env,stdout=log,stderr=subprocess.STDOUT,
                                   start_new_session=True)
        report['new_pid']=child.pid
        write_json(control/'handoff.json',report)
        ready=wait_for(new/'ready_for_resume.json',child,args.prepare_timeout,'prepare_corrected_baseline')
        if not ready.get('baseline_completed') or ready['anchor_sha256']!=cfg['checkpoint_sha256']:
            raise ValueError('Corrected original baseline was not completed')
        source=old/'checkpoint-progress.pth.tar'
        previous=source.stat();identity=(previous.st_ino,previous.st_mtime_ns)
        start=time.monotonic()
        print('기준 평가 완료. 다음 중간 저장 직후 전환합니다.',flush=True)
        while True:
            if child.poll() is not None:raise RuntimeError('Replacement exited before checkpoint handoff')
            process_identity(pid)
            current=source.stat()
            if (current.st_ino,current.st_mtime_ns)!=identity:break
            if time.monotonic()-start>args.checkpoint_timeout:raise TimeoutError('No new source checkpoint')
            time.sleep(.05)
        # Set the flag first so interruption after SIGSTOP still continues
        # the old trainer through rollback.
        paused=True;signal_original(pid,token,signal.SIGSTOP)
        report.update(status='source_suspended',paused_at=time.time())
        write_json(control/'handoff.json',report)
        snapshot=snapshot_checkpoint(source,resume)
        report.update(status='restoring',snapshot=snapshot)
        write_json(control/'handoff.json',report)
        restoration=wait_for(new/'resume_report.json',child,args.restore_timeout,'restore_model_and_optimizer')
        if (not restoration.get('weights_identical') or not restoration.get('optimizer_state_identical')
                or restoration['checkpoint_sha256']!=snapshot['sha256']
                or restoration['restored_global_step']!=snapshot['steps']):
            raise ValueError('Restoration evidence disagrees with immutable checkpoint')
        done=wait_for(new/'handoff_confirmed.json',child,args.restore_timeout,'first_resumed_update')
        saved=Path(done['checkpoint'])
        if (done['from_step']!=snapshot['steps'] or done['step']!=snapshot['steps']+1
                or saved.parent!=new or sha256(saved)!=done['checkpoint_sha256']):
            raise ValueError('First continued checkpoint was not verified')
        if child.poll() is not None:
            raise RuntimeError('Replacement exited after first-update checkpoint')
        confirmed=True
        report.update(status='resumed',resume=restoration,first_update=done,completed_at=time.time())
        write_json(control/'handoff.json',report)
        # The old trainer predates cooperative stop handlers. Its weights and
        # AdamW are now retained on disk and verified in the new trainer.
        signal_original(pid,token,signal.SIGTERM)
        try:signal_original(pid,token,signal.SIGCONT)
        except (ProcessLookupError,FileNotFoundError):pass
        paused=False
        report['old_process_termination_requested']=True
        write_json(control/'handoff.json',report)
        write_json(args.control_root.resolve()/'latest_handoff.json',report)
        print('전환 완료: '+str(new),flush=True)
        print('로그: '+str(log_path),flush=True)
        print('새 학습은 백그라운드에서 계속됩니다.',flush=True)
    except BaseException as exc:
        # Repeated Ctrl-C/SIGTERM must not interrupt liveness restoration.
        for sig in previous_handlers:signal.signal(sig,signal.SIG_IGN)
        report.update(status='error_after_confirmation' if confirmed else 'rolled_back',error=repr(exc))
        if not confirmed:
            try:rollback(child,pid,token,paused,report)
            except BaseException as cleanup_error:
                report['cleanup_error']=repr(cleanup_error)
        if confirmed and paused:
            try:
                signal_original(pid,token,signal.SIGTERM)
                signal_original(pid,token,signal.SIGCONT)
            except (ProcessLookupError,FileNotFoundError):pass
        write_json(control/'handoff.json',report)
        raise
    finally:
        for sig,handler in previous_handlers.items():signal.signal(sig,handler)


if __name__=='__main__':main()
