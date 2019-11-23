import json
from dcpower.testpower import data_preprocess

def build_cpu_metric(cpu_count):
    metric_name = ['idle', 'system', 'nice', 'steal', 'softirq', 'user', 'wait', 'interrupt']
    cpu_metric_list = []
    for i in range(cpu_count):
        print(i)
        cpu_metric_list += list(map(lambda x: 'cpu.%d.cpu.%s' % (i, x), metric_name))
    return cpu_metric_list

def calc_cpu_usage(data_cpu, cpu_count):
    data_cpu = list(map(lambda x:x.diff().iloc[1:, :], data_cpu))
    cpu_df = data_preprocess(data_cpu)
    for i, per_cpu_df in enumerate(cpu_df):
        per_cpu_df['cpu.%d.cpu.total' % (i)] = per_cpu_df.sum(axis=1)
        per_cpu_df['cpu.%d.cpu.total' % (i)] = 1 - per_cpu_df.iloc[:, 0] / per_cpu_df.iloc[:, -1]
        print(per_cpu_df.iloc[:, -1])
        if i == 0:
            cpu_all_df = per_cpu_df.iloc[:, -1]
        else:
            cpu_all_df += per_cpu_df.iloc[:, -1]
    cpu_all_df *= 100
    cpu_all_json = json.loads(cpu_all_df.to_json(orient='index'))
    #print(cpu_all_json)
    return cpu_all_json
