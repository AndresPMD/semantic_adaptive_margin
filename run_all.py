import subprocess
import argparse
import os

flatten = lambda t: [item for sublist in t for item in sublist]
softer = lambda out_metric: [[eval(i.split(': ')[1]) for i in e.split(',')[1:4]] for e in out_metric.split('\n') if e.startswith('Softer score')]
harder = lambda out_metric: [[eval(i.split(': ')[1]) for i in e.split(',')[1:4]] for e in out_metric.split('\n') if e.startswith('Hard score with Recall')]
print_func = lambda x: ' '.join('{:.2f}'.format(i) for i in flatten(x))

def get_metrics(args, sums, metric_name = 'cider'):
    metric_env_python = os.path.join(args.metric_env, 'bin/python')
    cmd = subprocess.Popen( metric_env_python + ' metrics/eval.py --metric_name '+ metric_name
                           +' --dataset '+ args.dataset
                           +' --recall_type recall --model_name '+ args.model_name
                           +' --split ' + args.split,
                           shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    out_metric, err_metric = cmd.communicate()
    out_metric = out_metric.decode('utf-8')
    m_results_softer = softer(out_metric)
    # m_results_hard = harder(out_metric)

    cmd = subprocess.Popen(metric_env_python + ' metrics/eval.py  --metric_name ' + metric_name
                           +' --dataset '+ args.dataset
                           +' --recall_type recall --include_anns --model_name ' + args.model_name
                           +' --split ' + args.split,
        shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out_metric, err_metric = cmd.communicate()
    out_metric = out_metric.decode('utf-8')

    m_results_softer_with_anns = softer(out_metric)
    m_results_hard_with_anns = harder(out_metric)

    print(metric_name.capitalize()+' -NON GT Softer:')
    print(print_func(m_results_softer))
    print(metric_name.capitalize()+' -GT Softer:')
    print(print_func(m_results_softer_with_anns))

    if metric_name=='cider':
        sums[5] = sum(flatten(m_results_hard_with_anns)[:3])
        sums[0] = sum(flatten(m_results_softer_with_anns))
        sums[1] = sum(flatten(m_results_softer))
    elif metric_name == 'spice':
        sums[2] = sum(flatten(m_results_softer_with_anns))
        sums[3] = sum(flatten(m_results_softer))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    #parser.add_argument('--model', default='/home/amafla/CVSE/runs/coco/CVSE_scratch_mini_10/model_best.pth.tar')
    parser.add_argument('--model', default='/data1slow/users/amafla/Adaptive_Margin/CVSE/runs/coco/CVSE_Cider5_RS_div6_scratch_tuning_mini_3/model_best.pth.tar')
    parser.add_argument('--model_env', default='/home/amafla/miniconda3/envs/pytorch12')
    parser.add_argument('--metric_env', default='/home/amafla/miniconda3/envs/pytorch12')
    parser.add_argument('--dataset', default='coco')
    #parser.add_argument('--model_name', default='sims_CVSE_cider_f30k_precomp.json')
    parser.add_argument('--model_name', default='CVSE_cider')
    parser.add_argument('--data_path', default='/data2fast/users/amafla/Precomp_features/data/')
    parser.add_argument('--vocab_path', default='/data1slow/users/amafla/Adaptive_Margin/CVSE/vocab/')
    parser.add_argument('--split', default='test', help='Choose to evaluate on coco 1k test set or 5k test set. (test | testall)')
    parser.add_argument('--transfer_test', action='store_true', help='Cross Eval Coco to Flickr30k')

    args = parser.parse_args()
    if args.dataset == 'f30k': data_name = 'f30k_precomp'
    elif args.dataset == 'coco': data_name = 'coco_precomp'

    sums = [0]*6
    if args.transfer_test:
        string_cmd = args.model_env+'/bin/python evaluate.py --model_path ' + args.model + ' --data_name f30k_precomp' + ' --data_name_vocab ' + data_name + ' --data_path ' + args.data_path + ' --vocab_path ' + args.vocab_path + ' --split ' + args.split + ' --transfer_test'

        cmd = subprocess.Popen(string_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        args.dataset = 'f30k'
    else:
        string_cmd = args.model_env+'/bin/python evaluate.py --model_path ' + args.model + ' --data_name ' + data_name + ' --data_name_vocab ' + data_name + ' --data_path ' + args.data_path + ' --vocab_path ' + args.vocab_path + ' --split ' + args.split
    
        cmd = subprocess.Popen(string_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
    print ('Command: ', string_cmd)
    out, err = cmd.communicate()
    #import pdb; pdb.set_trace()

    out = out.decode('utf-8')
    vse_recall = [i.split(':')[1].split(' ')[1:4] for i in out.split('\n') if i.startswith('Image to') or i.startswith('Text to')]
    vse_recall = [[eval(e) for e in elm] for elm in vse_recall]

    sums[4] = sum(flatten(vse_recall))
    print('Finished with model, calculating metrics')
    print('VSE RECALL:')
    print(print_func(vse_recall))
    get_metrics(args, sums, 'spice')
    get_metrics(args, sums, 'cider')
    print('Sums: Cider-GT, Cider-NonGT, Spice-GT, Spice-NonGt, VSE_recall, Recall')
    print(' '.join('{:.2f}'.format(i) for i in sums))
