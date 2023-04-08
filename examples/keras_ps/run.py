import subprocess
import json
import portpicker
from config import parse_args


def main():
    args = parse_args()
    PS = args.num_ps
    WORKERS = args.num_workers
    config_dict = {
        'cluster': {
            'ps': [f"localhost:{portpicker.pick_unused_port()}" for _ in range(PS)],
            'worker': [f"localhost:{portpicker.pick_unused_port()}" for _ in range(WORKERS)],
            'chief': [f"localhost:{portpicker.pick_unused_port()}"]
        },
        'task': {'type': 'chief', 'index': 0}
    }
    print(json.dumps(config_dict))
    procs = []
    for i in range(WORKERS):
        config_dict['task'] = {'type': 'worker', 'index': i}
        procs.append(subprocess.run(f"python3 main_ps_keras.py &", shell=True, env=dict(TF_CONFIG=json.dumps(config_dict), HIP_VISIBLE_DEVICES=str(4+i))))
    for i in range(PS):
        config_dict['task'] = {'type': 'ps', 'index': i}
        procs.append(subprocess.run(f"python3 main_ps_keras.py &", shell=True, env=dict(TF_CONFIG=json.dumps(config_dict), HIP_VISIBLE_DEVICES='-1')))

    config_dict['task'] = {'type': 'chief', 'index': 0}
    subprocess.run(f"python3 main_ps_keras.py --epochs={args.epochs} --batch_size={args.batch_size} --num_batch={args.num_batch} --embedding_dims={args.embedding_dims} --embedding_hash_size={args.embedding_hash_size}", shell=True, env=dict(TF_CONFIG=json.dumps(config_dict), HIP_VISIBLE_DEVICES='-1'))
    subprocess.run(f"kill -9 ```pidof python3 main_ps_keras.py```", shell=True)

main()