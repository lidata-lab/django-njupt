import requests
import json
import numpy as np
import pandas as pd
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from django.http import HttpRequest, HttpResponse

from dcpower.feadeal import build_cpu_metric, calc_cpu_usage
from dcpower.testpower import lasso_vm2pc_pred, lasso_vm2pc_train, getmax
from dcpower.dockerpower import build_docker_metrics, calc_docker_power
from dcpower.predpower import pred_preprocess, pcpower_pred_train, pcpower_pred
from dcpower.nnpower import nn_vm2pc_train, nn_vm2pc_pred, nn_toy

query_prefix = 'http://106.74.18.237:30763/api/query?'
#query_prefix = 'http://10.160.105.178:4242/api/query?'
# query_prefix = 'http://192.168.0.245:4242/api/query?'
put_prefix = 'http://106.74.18.237:30763/api/put?details'
#put_prefix = 'http://10.160.105.178:4242/api/put?details'
# put_prefix = 'http://192.168.0.245:4242/api/put?details'
metrics = ['cpu.usage.percent', 'memory.used.percent', 'interface.eth0.if_octets.rx', 'interface.eth0.if_octets.tx', 'disk.vda.disk_octets.write', 'disk.vda.disk_octets.read', 'pdu.power']
pc_metrics = ['cpu.active.percent', 'memory.used.percent', 'interface.eth0.if_octets.rx', 'interface.eth0.if_octets.tx', 'disk.sda.disk_octets.write', 'disk.sda.disk_octets.read', 'pdu.power']
cpu_count = 2
docker_num = 2

def test(request):
    cpu_metric_list = build_cpu_metric(cpu_count)
    start = request.POST.get('start')
    end = request.POST.get('end')
    print(start, end)
    host = request.POST.get('hostname')
    for i, metric in enumerate(cpu_metric_list):
        if i == 0:
            if end == None:
                query_midfix = 'start=%s&m=sum:%s{host=%s}' % (start, metric, host)
            else:
                query_midfix = 'start=%s&end=%s&m=sum:%s{host=%s}' % (start, end, metric, host)
        else:
            query_midfix = '%s&m=sum:%s{host=%s}' % (query_midfix, metric, host)
    print('%s%s' % (query_prefix, query_midfix))
    data_content = requests.get('%s%s' % (query_prefix, query_midfix))
    json_list = data_content.json()
    try:
        data_list = list(map(lambda x: x['dps'], json_list))
    except TypeError:
        return HttpResponse('Error:\nVM Quest Url Error!\nstart: %s\nend: %s\nhostname: %s' % (start, end, host))
    df_list = []
    for i, data_json in enumerate(data_list):
        if i % 8  == 0:
            df_list.append(pd.DataFrame.from_dict(data_json, orient='index', columns=[cpu_metric_list[i]]).sort_index())
        else:
            tmp_df = pd.DataFrame.from_dict(data_json, orient='index', columns=[cpu_metric_list[i]]).sort_index()
            df_list[int(i/8)] = (pd.concat([df_list[int(i/8)], tmp_df], axis=1))
    # pd.set_option('display.max_columns', None)
    cpu_usage_json = calc_cpu_usage(df_list, cpu_count)
    url2db = []
    session = requests.Session()
    retry = Retry(connect=5, backoff_factor=0.5)
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter)
    for i, ts in enumerate(cpu_usage_json.keys()):
        # url2db.append({"metric":"cpu.usage.percent","timestamp":ts,"value":cpu_usage_json[ts],"tags":{"host":host,"fqdn":host}})
        url2db.append({"metric": "cpu.usage.percent", "timestamp": ts, "value": cpu_usage_json[ts],
                       "tags": {"fqdn": host}})
        if i % 10 == 9 or (i % 10 != 9 and i == len(cpu_usage_json) - 1):
            try:
                session.post(put_prefix, json=url2db)
            except requests.exceptions.ConnectionError:
                session = requests.Session()
                session.mount('http://', adapter)
            url2db = []
    #print(url2db)
    #    return HttpResponse('Insert Error!')
    return HttpResponse('Insert Sucess!')

def dump_cpuactive(request):
    start = request.POST.get('start')
    end = request.POST.get('end')
    print(start, end)
    host = request.POST.get('hostname')
    query_midfix = 'start=%s&m=sum:%s{host=%s}' % (start, metric, host)
    print('%s%s' % (query_prefix, query_midfix))
    data_content = requests.get('%s%s' % (query_prefix, query_midfix))
    json_list = data_content.json()
    try:
        data_list = list(map(lambda x: x['dps'], json_list))
    except TypeError:
        return HttpResponse('Error:\nVM Quest Url Error!\nstart: %s\nend: %s\nhostname: %s' % (start, end, host))
    df_list = []
    for i, data_json in enumerate(data_list):
        if i % 8  == 0:
            df_list.append(pd.DataFrame.from_dict(data_json, orient='index', columns=[cpu_metric_list[i]]).sort_index())
        else:
            tmp_df = pd.DataFrame.from_dict(data_json, orient='index', columns=[cpu_metric_list[i]]).sort_index()
            df_list[int(i/8)] = (pd.concat([df_list[int(i/8)], tmp_df], axis=1))
    # pd.set_option('display.max_columns', None)
    cpu_usage_json = calc_cpu_usage(df_list, cpu_count)
    url2db = []
    session = requests.Session()
    retry = Retry(connect=5, backoff_factor=0.5)
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter)
    for i, ts in enumerate(cpu_usage_json.keys()):
        url2db.append({"metric":"cpu.usage.percent","timestamp":ts,"value":cpu_usage_json[ts],"tags":{"host":host,"fqdn":host}})
        if i % 10 == 9 or (i % 10 != 9 and i == len(cpu_usage_json) - 1):
            try:
                session.post(put_prefix, json=url2db)
            except requests.exceptions.ConnectionError:
                session = requests.Session()
                session.mount('http://', adapter)
            url2db = []
    #print(url2db)
    #    return HttpResponse('Insert Error!')
    return HttpResponse('Insert Sucess!')

# VM forcasting
def from_mainsys(request):
    print(request)
    # Analyse POST request
    start = request.POST.get('start')
    end = request.POST.get('end')
    print(start, end)
    host = request.POST.get('hostname')
    pdunit = request.POST.get('pdu')
    train = request.POST.get('train')
    model = request.POST.get('model')
    # Build data requester
    for i, metric in enumerate(metrics[:-1]):
        if i == 0:
            if end == None:
                query_midfix = 'start=%s&m=sum:%s{host=%s}' % (start, metric, host)
            else:
                query_midfix = 'start=%s&end=%s&m=sum:%s{host=%s}' % (start, end, metric, host)
        elif i > 1:
            query_midfix = '%s&m=sum:%s{host=%s}' % (query_midfix, metric, host)
        else:
            query_midfix = '%s&m=sum:%s{host=%s}' % (query_midfix, metric, host)
    if pdunit != None:
        query_midfix = '%s&m=sum:%s{host=%s}' % (query_midfix, metrics[-1], pdunit)
    print('%s%s' % (query_prefix, query_midfix))
    data_content = requests.get('%s%s' % (query_prefix, query_midfix))
    json_list = data_content.json()
    try:
        data_list = list(map(lambda x: x['dps'], json_list))
    except TypeError:
        return HttpResponse('Error:\nVM Quest Url Error!\nstart: %s\nend: %s\nhostname: %s\npdu: %s' % (start, end, host, pdunit))
    host_list = host.split('|')
    hostnum = len(host_list)
    print(hostnum)
    df_list = []
    # Single Calc
    if hostnum == 1:
        for i, data_json in enumerate(data_list):
            if i == 0:
                df_list.append(pd.DataFrame.from_dict(data_json, orient='index', columns=[metrics[i]]).sort_index())
            else:
                tmp_df = pd.DataFrame.from_dict(data_json, orient='index', columns=[metrics[i]]).sort_index()
                df_list[0] = (pd.concat([df_list[0], tmp_df], axis=1))         
        power_pred_json = lasso_vm2pc_pred(df_list)
        return HttpResponse(json.dumps(power_pred_json))
    print((len(metrics) - 1) * hostnum, len(data_list))
    if (len(metrics) - 1) * hostnum != len(data_list) - 1:
        return HttpResponse('Error:\nHost num Error!\nQuery Host num: %d\nGet Host num: %d' % (hostnum, (len(data_list) - 1) / (len(metrics) - 1)))
    # Multi calc
    for i, data_json in enumerate(data_list):
        if i >= len(data_list[:-1]):
            power_df = pd.DataFrame.from_dict(data_list[-1], orient='index', columns=[metrics[-1]]).sort_index()
        elif i == 0 or i == 1:
            df_list.append(pd.DataFrame.from_dict(data_json, orient='index', columns=[metrics[int(i/2)]]).sort_index())
        else:
            tmp_df = pd.DataFrame.from_dict(data_json, orient='index', columns=[metrics[int(i/2)]]).sort_index()
            df_list[i%2] = (pd.concat([df_list[i%2], tmp_df], axis=1))
    #print(df_list)
    # df_list[0].to_csv('/home/fc10382/Mcoder/PowerEff/Data/tsdb/1in1_idle/vm1.csv')
    # df_list[1].to_csv('/home/fc10382/Mcoder/PowerEff/Data/tsdb/1in1_idle/vm2.csv')
    # # power_df.to_csv('/home/fc10382/Mcoder/PowerEff/Data/tsdb/1in1/power.csv')
    # return HttpResponse('Export Finish~')
    #print(power_df)
    #if model == 'linear1':
    #    if train == None:
    #        power_pred_json = nn_vm2pc_pred(df_list, power_df)
    #        return HttpResponse(json.dumps(power_pred_json))
    #    else:
    #        nn_vm2pc_train(df_list, power_df)
    #        return HttpResponse('Train Sucess!')
    #        # y_true, y_pred = nn_toy(df_list, power_df)
    #        # print(y_true, y_pred)
    #        # return HttpResponse(plot2line(np.squeeze(y_true), np.squeeze(y_pred)).render_embed())
    if train == None:
        power_pred_json = lasso_vm2pc_pred(df_list, power_df)
        return HttpResponse(json.dumps(power_pred_json))
    else:
        #getmax(df_list)
        lasso_vm2pc_train(df_list, power_df)
        return HttpResponse('Train Finished!')
    #return HttpResponse('Success!')

def docker_percent(request):
    print(request)
    # Analyse POST request
    start = request.POST.get('start')
    end = request.POST.get('end')
    print(start, end)
    host = request.POST.get('hostname')
    vmno = host.replace('openstack0', '')
    docker_metrics_list = build_docker_metrics(vmno, docker_num)
    #print(docker_metrics_list)
    # Build data requester
    for i, metric in enumerate(metrics[:-1]):
        if i == 0:
            if end == None:
                query_midfix = 'start=%s&m=sum:%s{host=%s}' % (start, metric, host)
            else:
                query_midfix = 'start=%s&end=%s&m=sum:%s{host=%s}' % (start, end, metric, host)
        elif i > 1:
            query_midfix = '%s&m=sum:%s{host=%s}' % (query_midfix, metric, host)
        else:
            query_midfix = '%s&m=sum:%s{host=%s}' % (query_midfix, metric, host)
    for j in range(docker_num):
        for metric in docker_metrics_list[j]:
            query_midfix = '%s&m=sum:%s' % (query_midfix, metric)
    print('%s%s' % (query_prefix, query_midfix))
    data_content = requests.get('%s%s' % (query_prefix, query_midfix))
    json_list = data_content.json()
    try:
        data_list = list(map(lambda x: x['dps'], json_list))
        #print(data_list)
    except TypeError:
        return HttpResponse('Error:\nDocker Quest Url Error!\nstart: %s\nend: %s\nhostname: %s\n' % (start, end, host))
    df_list = []
    power_list = []
    for i, data_json in enumerate(data_list):
        if i%6 == 0:
            data_df = pd.DataFrame.from_dict(data_json, orient='index', columns=[metrics[i%6]]).sort_index()
        else:
            tmp_df = pd.DataFrame.from_dict(data_json, orient='index', columns=[metrics[i%6]]).sort_index()
            data_df = (pd.concat([data_df, tmp_df], axis=1))
        if i%6 == 5:
            df_list.append(data_df)
    # df_list[1].to_csv('/home/fc10382/Mcoder/PowerEff/Data/tsdb/1in1/docker20.csv')
    # # df_list[2].to_csv('/home/fc10382/Mcoder/PowerEff/Data/tsdb/1in1/docker2.csv')
    # return HttpResponse('Export Finish~')
    power_list = calc_docker_power(df_list, vmno)
    return HttpResponse(json.dumps(power_list))
    
def pred_power(request):
    his_len = 30
    print(request)
    # Analyse POST request
    start = request.POST.get('start')
    host = request.POST.get('hostname')
    pdunit = request.POST.get('pdu')
    train = request.POST.get('train')
    time_unit = request.POST.get('timeunit')
    if pdunit == None:
        return HttpResponse('arg pdu must be set!')
    if time_unit == None:
        time_unit = 15
    else:
        time_unit = int(time_unit)
    if train != None:
        #his_len = 30 + 2
        #his_len = 8875
        his_len = 26625
        # his_len = 53250
    end = int(start) - 1
    start = int(start) - time_unit * his_len
    print(start, end)
    for i, metric in enumerate(pc_metrics[:-1]):
        if i == 0:
            query_midfix = 'start=%s&end=%s&m=sum:%s{fqdn=%s}' % (start, end, metric, host)
        elif i > 1:
            query_midfix = '%s&m=sum:%s{fqdn=%s}' % (query_midfix, metric, host)
        else:
            query_midfix = '%s&m=sum:%s{fqdn=%s}' % (query_midfix, metric, host)
    query_midfix = '%s&m=sum:%s{host=%s}' % (query_midfix, pc_metrics[-1], pdunit)
    print('%s%s' % (query_prefix, query_midfix))
    data_content = requests.get('%s%s' % (query_prefix, query_midfix))
    json_list = data_content.json()
    try:
        data_list = list(map(lambda x: x['dps'], json_list))
    except TypeError:
        return HttpResponse('Error:\nPred Quest Url Error!\nstart: %s\nend: %s\nhostname: %s\npdu: %s' % (start, end, host, pdunit))
    for i, data_json in enumerate(data_list):
        if i >= len(data_list[:-1]):
            power_df = pd.DataFrame.from_dict(data_list[-1], orient='index', columns=[pc_metrics[-1]]).sort_index()
        elif i == 0:
            fea_df = pd.DataFrame.from_dict(data_json, orient='index', columns=[pc_metrics[i]]).sort_index()
        else:
            tmp_df = pd.DataFrame.from_dict(data_json, orient='index', columns=[pc_metrics[i]]).sort_index()
            fea_df = (pd.concat([fea_df, tmp_df], axis=1))         
    print(fea_df)
    # fea_df.to_csv('/home/fc10382/Mcoder/PowerEff/Data/tsdb/1in1_idle/pc.csv')
    # return HttpResponse('Export Finish~')
    print(power_df)
    if train == None:
        y_true, y_np = pcpower_pred(fea_df, power_df, time_unit)
        #print(type(y_true), type(y_np))
        return HttpResponse(json.dumps([y_true, y_np]))
    else:
        pcpower_pred_train(fea_df, power_df, time_unit)
        return HttpResponse('Train Success!')
    
def pcvm_power(request):
    his_len = 30
    print(request.POST)
    # Analyse POST request
    start = request.POST.get('start')
    host = request.POST.get('hostname')
    pdunit = request.POST.get('pdu')
    train = request.POST.get('train')
    time_unit = request.POST.get('timeunit')
    if pdunit == None:
        return HttpResponse('arg pdu must be set!')
    if time_unit == None:
        time_unit = 10
    else:
        time_unit = int(time_unit)
    if train != None:
        #his_len = 30 + 2
        #his_len = 8875
        #his_len = 26625
        his_len = 53250
    end = int(start) - 1
    start = int(start) - time_unit * his_len
    print(start, end)
    for i, metric in enumerate(pc_metrics[:-1]):
        if i == 0:
            query_midfix = 'start=%s&end=%s&m=sum:%s{fqdn=%s}' % (start, end, metric, host)
        elif i > 1:
            query_midfix = '%s&m=sum:%s{fqdn=%s}' % (query_midfix, metric, host)
        else:
            query_midfix = '%s&m=sum:%s{fqdn=%s}' % (query_midfix, metric, host)
    query_midfix = '%s&m=sum:%s{host=%s}' % (query_midfix, pc_metrics[-1], pdunit)
    print('%s%s' % (query_prefix, query_midfix))
    data_content = requests.get('%s%s' % (query_prefix, query_midfix))
    json_list = data_content.json()
    try:
        data_list = list(map(lambda x: x['dps'], json_list))
    except TypeError:
        return HttpResponse('Error:\nPred Quest Url Error!\nstart: %s\nend: %s\nhostname: %s\npdu: %s' % (start, end, host, pdunit))
    for i, data_json in enumerate(data_list):
        if i >= len(data_list[:-1]):
            power_df = pd.DataFrame.from_dict(data_list[-1], orient='index', columns=[pc_metrics[-1]]).sort_index()
        elif i == 0:
            fea_df = pd.DataFrame.from_dict(data_json, orient='index', columns=[pc_metrics[i]]).sort_index()
        else:
            tmp_df = pd.DataFrame.from_dict(data_json, orient='index', columns=[pc_metrics[i]]).sort_index()
            fea_df = (pd.concat([fea_df, tmp_df], axis=1))         
    print(fea_df)
    print(power_df)
    if train == None:
        y_true, y_np = pcpower_pred(fea_df, power_df, time_unit)
        #print(type(y_true), type(y_np))
        vm_y = lasso_vm2pc_pred(fea_df.iloc[-time_unit:, :])
        return HttpResponse(json.dumps([y_true, y_np]+vm_y))
    else:
        pcpower_pred_train(fea_df, power_df, time_unit)
        return HttpResponse('Train Success!')

def cpu_freq(request):
    cpu_count = 4
    print(request)
    start = request.POST.get('start')
    end = request.POST.get('end')
    print(start, end)
    host = request.POST.get('hostname')
    for i in range(cpu_count):
        if i == 0:
            query_midfix = 'start=%s&end=%s&m=sum:cpufreq.%d.cpufreq{fqdn=%s}' % (start, end, i, host)
        else:
            query_midfix = '%s&m=sum:cpufreq.%d.cpufreq{fqdn=%s}' % (query_midfix, i, host)
    print('%s%s' % (query_prefix, query_midfix))
    data_content = requests.get('%s%s' % (query_prefix, query_midfix))
    json_list = data_content.json()
    data_list = list(map(lambda x: x['dps'], json_list))
    for i, data_json in enumerate(data_list):
        if i == 0:
            fea_df = pd.DataFrame.from_dict(data_json, orient='index', columns=['cpu.'+str(i)+'.freq']).sort_index()
        else:
            tmp_df = pd.DataFrame.from_dict(data_json, orient='index', columns=['cpu.'+str(i)+'.freq']).sort_index()
            fea_df = (pd.concat([fea_df, tmp_df], axis=1))
    print(fea_df)
    fea_df.to_csv('/home/fc10382/Mcoder/PowerEff/Data/tsdb/stress/freq.csv')
    return HttpResponse('Export Success!')
