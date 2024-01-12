from retro_star.api import RSPlanner
import sys


def worker(q, thread_id, args, start_idx, end_idx):
    msg = run_multistep(thread_id, args, start_idx, end_idx)
    q.put(msg)


def listener(q, save_name):
    with open(save_name, 'a') as f:
        while True:
            m = q.get()
            if m == '#done#':
                break
            f.write(str(m))
            f.flush()


def run_multistep(thread_id, args, mol_ids, products):
    print(f'[Thread {thread_id}] start')
    planner = RSPlanner(
        cuda=args.device=='cuda',
        iterations=args.iterations,
        expansion_topk=args.exp_topk,
        route_topk=args.route_topk,
        beam_size=args.beam_size,
        model_type=args.model_type,
        model_path=args.model_path,
        retrieval=args.retrieval=='true',
        retrieval_db=args.retrieval_db,
        path_retrieval=args.path_retrieval=='true',
        path_retrieval_db=args.path_retrieval_db,
        starting_molecules=args.blocks
    )
    msg = ''
    done = 0
    for mol_id, product in zip(mol_ids, products):
        try:
            result = planner.plan(product)
        except KeyboardInterrupt:
            print('KeyboardInterrupt')
            sys.exit()
        except:
            result = 'Error'

        if result is None:
            msg += f'{mol_id} None\n'
        elif result == 'Error':
            msg += f'{mol_id} Error\n'
        else:
            for i, route in enumerate(result):
                msg += f'{mol_id} {i} {route}\n'
        done += 1
        print(f'[Thread {thread_id}] {done}/{len(products)}')
    print(f'[Thread {thread_id}] Finished')
    return msg
